param(
    [string]$RunDate = (Get-Date).ToString("yyyy-MM-dd"),
    [switch]$SkipAppend,
    [switch]$DryRun,
    [switch]$Apply,
    [switch]$ReloadLocalBackend,
    [string]$AdminReloadUrl = "http://127.0.0.1:8000/api/v1/admin/reload",
    [string]$RunDir = "docs\cp212_unified_refresh_runs"
)

$ErrorActionPreference = "Stop"

if ($DryRun -and $Apply) {
    throw "-DryRun과 -Apply는 동시에 사용할 수 없습니다."
}
if (-not $DryRun -and -not $Apply) {
    throw "운영 모드는 -Apply 또는 -DryRun 중 하나를 명시해야 합니다."
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot "serving_notification_status.ps1")

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SafeDate = $RunDate.Replace("-", "")
$RunDirPath = Join-Path $Root $RunDir
$LogDir = Join-Path $Root "logs\cp212_unified_refresh"
New-Item -ItemType Directory -Force -Path $RunDirPath | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$PipelineLogPath = Join-Path $LogDir "pipeline_${SafeDate}_${RunStamp}.log"
$MetricsPath = Join-Path $RunDirPath "cp212_unified_refresh_metrics_${SafeDate}_${RunStamp}.json"
$ReportPath = Join-Path $RunDirPath "cp212_unified_refresh_report_${SafeDate}_${RunStamp}.md"
$LatestMetricsPath = Join-Path $Root "docs\cp212_integration_metrics.json"
$LatestReportPath = Join-Path $Root "docs\cp212_integration_report.md"
$LatestSchedulePath = Join-Path $Root "docs\cp212_schedule_status.md"

$env:MARKET_DATA_PROVIDER = "yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER = ""
$env:EODHD_API_KEY = ""
$env:LENS_DATA_BACKEND = "local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS = "1"
$env:LENS_LOCAL_SNAPSHOT_DIR = Join-Path $Root "data\parquet"
$env:WANDB_MODE = "disabled"
$env:YFINANCE_FETCH_METHOD = "direct_chart"
$env:PYTHONUTF8 = "1"

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}
$ModeArg = if ($Apply) { "--apply" } else { "--dry-run" }
$ModeText = if ($Apply) { "Apply" } else { "DryRun" }
$Steps = New-Object System.Collections.Generic.List[object]

function Write-Log {
    param([string]$Message)
    $line = "[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $Message
    Write-Host $line
    Add-Content -Path $PipelineLogPath -Value $line -Encoding UTF8
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath
    )
    $Started = Get-Date
    Write-Log "$Name 시작"
    $ExitCode = 0
    $Status = "PASS"
    # CP256 — PS 5.1 quirk 회피: ErrorActionPreference=Stop 하에서 native 프로세스가
    # stderr 로 한 줄이라도 쓰면 PowerShell 이 이를 terminating error(NativeCommandError)
    # 로 바꿔 catch 로 떨어뜨린다. yfinance 의 일시적 티커 경고(stderr) 때문에 append 가
    # 오탐 FAIL 나서 일일 refresh 가 4일 멈췄다. 이 함수 안에서는 Continue 로 낮추고,
    # 성공/실패는 오직 프로세스 종료코드($LASTEXITCODE)로만 판정한다.
    $PreviousEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Python @Arguments 1> $StdoutPath 2> $StderrPath
        $ExitCode = if ($LASTEXITCODE -ne $null) { [int]$LASTEXITCODE } else { 0 }
        if ($ExitCode -ne 0) {
            $Status = "FAIL"
        }
    } catch {
        $ExitCode = 1
        $Status = "FAIL"
        Add-Content -Path $StderrPath -Value $_.Exception.Message -Encoding UTF8
    } finally {
        $ErrorActionPreference = $PreviousEap
    }
    $Elapsed = [math]::Round(((Get-Date) - $Started).TotalSeconds, 3)
    Write-Log "$Name 종료: status=$Status exit_code=$ExitCode elapsed=${Elapsed}s"
    $Steps.Add([pscustomobject]@{
        name = $Name
        status = $Status
        exit_code = $ExitCode
        elapsed_seconds = $Elapsed
        stdout = $StdoutPath
        stderr = $StderrPath
    }) | Out-Null
    return $ExitCode
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

# CP256 — 일일 refresh 결과를 Slack 으로 알림. 환경변수 LENS_SLACK_WEBHOOK 이 설정된
# 경우에만 동작하며(없으면 조용히 skip), 실패해도 refresh 에 영향 주지 않는다.
# webhook URL 은 비밀이라 코드/깃에 넣지 않고 환경변수로만 읽는다.
function Send-SlackNotification {
    param(
        [string]$Text,
        [string]$Color = "good"
    )
    $webhook = $env:LENS_SLACK_WEBHOOK
    if (-not $webhook) {
        # setx 로 저장한 사용자 환경변수를 레지스트리에서 직접 읽는 fallback.
        # 프로세스가 setx 이전에 떠서 상속 못 받은 경우(예약 작업 포함) 대비 — 로그오프/재시작 불필요.
        try { $webhook = [Environment]::GetEnvironmentVariable("LENS_SLACK_WEBHOOK", "User") } catch { }
    }
    if (-not $webhook) { return "SKIPPED_NO_WEBHOOK" }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $payload = @{
            attachments = @(@{ color = $Color; text = $Text; mrkdwn_in = @("text") })
        } | ConvertTo-Json -Depth 6
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
        Invoke-RestMethod -Uri $webhook -Method Post -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 15 | Out-Null
        return "SENT"
    } catch {
        return "SEND_FAILED"
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

Write-Log "CP212 unified v1 refresh 시작: mode=$ModeText run_date=$RunDate skip_append=$SkipAppend"

$AppendMetricsPath = Join-Path $RunDirPath "append_metrics_${SafeDate}_${RunStamp}.json"
$AppendReportPath = Join-Path $RunDirPath "append_report_${SafeDate}_${RunStamp}.md"
if (-not $SkipAppend) {
    $AppendArgs = @(
        "scripts\cp151_yfinance_500_backfill.py",
        "--start-date", "2015-01-01",
        "--end-date", $RunDate,
        "--chunk-size", "75",
        "--max-chunks-per-run", "8",
        "--sleep-seconds-between-tickers", "0.2",
        "--sleep-seconds-between-chunks", "30",
        "--fetch-mode", "incremental",
        "--incremental-lookback-days", "10",
        "--indicator-mode", "auto",
        "--metrics-path", $AppendMetricsPath,
        "--report-path", $AppendReportPath,
        "--failed-tickers-csv", (Join-Path $RunDirPath "append_failed_tickers_${SafeDate}_${RunStamp}.csv"),
        "--latest-distribution-csv", (Join-Path $RunDirPath "append_latest_distribution_${SafeDate}_${RunStamp}.csv")
    )
    $AppendArgs += $ModeArg
    Invoke-PythonStep -Name "append" -Arguments $AppendArgs -StdoutPath (Join-Path $RunDirPath "append_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "append_stderr_${SafeDate}_${RunStamp}.log") | Out-Null
} else {
    Write-Log "append 단계 skip"
    $Steps.Add([pscustomobject]@{ name = "append"; status = "SKIPPED"; exit_code = $null; elapsed_seconds = 0; stdout = $null; stderr = $null }) | Out-Null
}

if ($Apply) {
    Invoke-PythonStep -Name "market_snapshot" -Arguments @("backend\scripts\build_v1_market_local.py", "--asof-date", $RunDate) -StdoutPath (Join-Path $RunDirPath "market_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "market_stderr_${SafeDate}_${RunStamp}.log") | Out-Null
} else {
    Write-Log "market snapshot은 dry-run에서 파일을 쓰므로 실행하지 않음"
    $Steps.Add([pscustomobject]@{ name = "market_snapshot"; status = "SKIPPED_DRY_RUN_WRITES_OUTPUT"; exit_code = $null; elapsed_seconds = 0; stdout = $null; stderr = $null }) | Out-Null
}

$BandMetricsPath = Join-Path $RunDirPath "band_refresh_metrics_${SafeDate}_${RunStamp}.json"
$BandReportPath = Join-Path $RunDirPath "band_refresh_report_${SafeDate}_${RunStamp}.md"
Invoke-PythonStep -Name "band_refresh" -Arguments @(
    "backend\scripts\cp210_band_forward_refresh.py",
    $ModeArg,
    "--batch-size", "2048",
    "--metrics-path", $BandMetricsPath,
    "--report-path", $BandReportPath
) -StdoutPath (Join-Path $RunDirPath "band_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "band_stderr_${SafeDate}_${RunStamp}.log") | Out-Null

$LineMetricsPath = Join-Path $RunDirPath "line_refresh_metrics_${SafeDate}_${RunStamp}.json"
$LineReportPath = Join-Path $RunDirPath "line_refresh_report_${SafeDate}_${RunStamp}.md"
Invoke-PythonStep -Name "line_refresh" -Arguments @(
    "backend\scripts\cp212_line_1d_export.py",
    $ModeArg,
    "--device", "auto",
    "--metrics-path", $LineMetricsPath,
    "--report-path", $LineReportPath
) -StdoutPath (Join-Path $RunDirPath "line_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "line_stderr_${SafeDate}_${RunStamp}.log") | Out-Null

if ($Apply) {
    Invoke-PythonStep -Name "product_history_rebuild" -Arguments @("backend\scripts\rebuild_product_history_parquet.py") -StdoutPath (Join-Path $RunDirPath "history_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "history_stderr_${SafeDate}_${RunStamp}.log") | Out-Null
    Invoke-PythonStep -Name "ai_runs_mock_rebuild" -Arguments @("backend\scripts\build_ai_runs_mock.py") -StdoutPath (Join-Path $RunDirPath "ai_runs_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "ai_runs_stderr_${SafeDate}_${RunStamp}.log") | Out-Null
    # CP246 — serving parquet 재인코딩(compaction). build 단계들은 문자열/날짜를
    # plain object 로 쓰므로, 그대로 두면 cold load read 스파이크가 누적돼 512MB
    # Render 무료에서 OOM(503/502)이 재발한다. 여기서 category(dictionary)+datetime
    # 으로 재저장해 read 스파이크를 제거한다. 스크립트가 값 동일성 게이트를 거쳐
    # 불일치 시 해당 파일을 건너뛰고 exit 1 → 아래 FailedSteps 가 push 를 막는다.
    # $FailedSteps 집계(아래) 전에 위치해야 compact 실패가 push 차단에 반영된다.
    Invoke-PythonStep -Name "parquet_compact" -Arguments @("backend\scripts\compact_v1_parquets.py", "--apply") -StdoutPath (Join-Path $RunDirPath "compact_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "compact_stderr_${SafeDate}_${RunStamp}.log") | Out-Null
} else {
    Write-Log "product history와 ai_runs_mock은 dry-run에서 파일을 쓰므로 실행하지 않음"
    $Steps.Add([pscustomobject]@{ name = "product_history_rebuild"; status = "SKIPPED_DRY_RUN_WRITES_OUTPUT"; exit_code = $null; elapsed_seconds = 0; stdout = $null; stderr = $null }) | Out-Null
    $Steps.Add([pscustomobject]@{ name = "ai_runs_mock_rebuild"; status = "SKIPPED_DRY_RUN_WRITES_OUTPUT"; exit_code = $null; elapsed_seconds = 0; stdout = $null; stderr = $null }) | Out-Null
}

$ReloadStatus = "SKIPPED"
if ($ReloadLocalBackend) {
    $Token = [Environment]::GetEnvironmentVariable("LENS_ADMIN_RELOAD_TOKEN")
    if (-not $Token) {
        $ReloadStatus = "SKIPPED_TOKEN_MISSING"
        Write-Log "admin reload token이 없어 local backend reload를 건너뜀"
    } else {
        try {
            Invoke-RestMethod -Method Post -Uri $AdminReloadUrl -Headers @{ "X-Lens-Admin-Token" = $Token } | Out-Null
            $ReloadStatus = "PASS"
            Write-Log "local backend unified reload 완료"
        } catch {
            $ReloadStatus = "WARN_RELOAD_FAILED"
            Write-Log "local backend unified reload 실패: $($_.Exception.Message)"
        }
    }
}

$BandMetrics = Read-JsonFile -Path $BandMetricsPath
$LineMetrics = Read-JsonFile -Path $LineMetricsPath
$AppendMetrics = Read-JsonFile -Path $AppendMetricsPath
$FailedSteps = @($Steps | Where-Object { $_.status -eq "FAIL" })
$AppendIsPartial = $false
if ($AppendMetrics -and $AppendMetrics.final_status -and ([string]$AppendMetrics.final_status).StartsWith("PARTIAL")) {
    $AppendIsPartial = $true
}
$ReloadIsPartial = $false
if ($ReloadLocalBackend -and $ReloadStatus -ne "PASS") {
    $ReloadIsPartial = $true
}
$FinalStatus = if ($FailedSteps.Count -eq 0 -and -not $AppendIsPartial -and -not $ReloadIsPartial) {
    "PASS_UNIFIED_REFRESH_ALIGNED"
} else {
    "WARN_UNIFIED_REFRESH_PARTIAL"
}
if ($ModeText -eq "DryRun") {
    $FinalStatus = if ($FailedSteps.Count -eq 0 -and -not $AppendIsPartial -and -not $ReloadIsPartial) {
        "PASS_UNIFIED_REFRESH_DRY_RUN"
    } else {
        "WARN_UNIFIED_REFRESH_DRY_RUN_PARTIAL"
    }
}

$Metrics = [pscustomobject]@{
    cp = "CP212-LG"
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    mode = $ModeText
    run_date = $RunDate
    final_status = $FinalStatus
    steps = $Steps
    reload_status = $ReloadStatus
    append_metrics_path = $AppendMetricsPath
    band_metrics_path = $BandMetricsPath
    line_metrics_path = $LineMetricsPath
    report_path = $ReportPath
    append_summary = $AppendMetrics
    band_summary = $BandMetrics
    line_summary = $LineMetrics
    append_is_partial = $AppendIsPartial
    reload_is_partial = $ReloadIsPartial
    forbidden_actions_observed = [pscustomobject]@{
        supabase_write = $false
        db_write = $false
        new_training = $false
        new_calibration = $false
        inference_training = $false
        line_1w_generation = $false
    }
}

$Metrics | ConvertTo-Json -Depth 20 | Set-Content -Path $MetricsPath -Encoding UTF8
Copy-Item -LiteralPath $MetricsPath -Destination $LatestMetricsPath -Force

$ReportLines = @(
    "# CP212 통합 refresh 실행 보고",
    "",
    "- final_status: ``$FinalStatus``",
    "- mode: ``$ModeText``",
    "- run_date: ``$RunDate``",
    "- reload_status: ``$ReloadStatus``",
    "- append_is_partial: ``$AppendIsPartial``",
    "- reload_is_partial: ``$ReloadIsPartial``",
    "",
    "## 단계 결과",
    "",
    "| 단계 | 상태 | exit_code | 소요초 |",
    "|---|---|---:|---:|"
)
foreach ($Step in $Steps) {
    $ReportLines += "| $($Step.name) | $($Step.status) | $($Step.exit_code) | $($Step.elapsed_seconds) |"
}
$ReportLines += @(
    "",
    "## 산출물",
    "",
    "- metrics: ``$MetricsPath``",
    "- band metrics: ``$BandMetricsPath``",
    "- line metrics: ``$LineMetricsPath``",
    "",
    "## 정책",
    "",
    "- Supabase/DB write 없음",
    "- 새 학습 없음",
    "- 새 calibration 없음",
    "- 1W line 생성 없음",
    "- line은 CP212 F4 beta=4 ensemble checkpoint로 최근 구간을 다시 계산해 serving parquet를 갱신"
)
$ReportLines -join "`n" | Set-Content -Path $ReportPath -Encoding UTF8
Copy-Item -LiteralPath $ReportPath -Destination $LatestReportPath -Force

$ScheduleLines = @(
    "# CP212 schedule status",
    "",
    "- updated_at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss KST')",
    "- runner: scripts/run_v1_unified_refresh_local.ps1",
    "- mode: $ModeText",
    "- status: $FinalStatus",
    "- reload_status: $ReloadStatus",
    "- metrics: $MetricsPath",
    "- report: $ReportPath"
)
$ScheduleLines -join "`n" | Set-Content -Path $LatestSchedulePath -Encoding UTF8

# 운영 데이터는 Git 이력이나 Supabase에 쓰지 않는다. 비공개 GitHub 릴리스에
# 검증 스냅샷을 발행한 뒤 Render의 인증 동기화 API를 호출한다. 최신 포인터를
# 마지막에 교체하고 최신·이전 첨부파일만 남긴다. 업로드 실패 시 기존 버전을 유지한다.
$PushStatus = "DISABLED_GITHUB_RELEASE"
$PublishStatus = "SKIPPED"
$PublishResultPath = Join-Path $RunDirPath "github_publish_${SafeDate}_${RunStamp}.json"
$ExpectedSnapshotId = ""
if ($Apply -and $FailedSteps.Count -eq 0) {
    Write-Log "github_publish 시작"
    $PublishLog = Join-Path $RunDirPath "github_publish_${SafeDate}_${RunStamp}.log"
    $PublishErrorLog = Join-Path $RunDirPath "github_publish_stderr_${SafeDate}_${RunStamp}.log"
    $PrevPyEnc = $env:PYTHONIOENCODING
    $env:PYTHONIOENCODING = "utf-8"
    $PrevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Python "backend\scripts\publish_serving_to_github.py" `
            --data-dir (Join-Path $Root "backend\data\v1") `
            --sync-mode periodic `
            --result-path $PublishResultPath 1> $PublishLog 2> $PublishErrorLog
        $PublishExit = if ($LASTEXITCODE -ne $null) { [int]$LASTEXITCODE } else { 0 }
    } catch {
        $PublishExit = 1
        Add-Content -Path $PublishErrorLog -Value $_.Exception.Message -Encoding UTF8
    } finally {
        $ErrorActionPreference = $PrevEap
        $env:PYTHONIOENCODING = $PrevPyEnc
    }
    if ($PublishExit -eq 0) {
        $PublishStatus = "PASS"
    } else {
        $PublishStatus = "FAIL_GITHUB_PUBLISH"
        Write-Log "CRITICAL: GitHub 릴리스 발행 또는 Render 동기화 실패(exit=$PublishExit). 현재 운영 버전은 유지된다. 로그 확인: $PublishLog, $PublishErrorLog"
    }
    try {
        $PublishResult = Read-JsonFile -Path $PublishResultPath
        if ($PublishResult -and $PublishResult.snapshot_id) {
            $ExpectedSnapshotId = [string]$PublishResult.snapshot_id
        }
    } catch {
        Write-Log "GitHub 발행 결과 JSON을 읽지 못함. 로그와 배포 상태를 확인한다."
    }
    Write-Log "github_publish 종료: status=$PublishStatus exit=$PublishExit"
} elseif ($Apply) {
    $PublishStatus = "SKIPPED_STEP_FAILURE"
    Write-Log "github_publish skip: 앞 단계 실패로 발행을 건너뜀"
} else {
    $PublishStatus = "SKIPPED_DRY_RUN"
}

# 발행 실패 때도 정상 원격 버전, 고정 폴백, 오래된 데이터와 응답 불가를 구분한다.
# 고정 폴백 정상 응답은 서비스 유지 경고이며 새 스냅샷 발행 성공으로 처리하지 않는다.
$VerifyStatus = "SKIPPED"
$VerifyDetail = ""
$VerifyData = $null
$VerifyResultPath = Join-Path $RunDirPath "deployed_verify_${SafeDate}_${RunStamp}.json"
if ($Apply) {
    Write-Log "deployed_verify 시작"
    $VerifyLog = Join-Path $RunDirPath "deployed_verify_${SafeDate}_${RunStamp}.log"
    $PrevPyEnc2 = $env:PYTHONIOENCODING
    $env:PYTHONIOENCODING = "utf-8"
    $PrevEap2 = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $VerifyArguments = @("backend\scripts\verify_deployed_serving.py", "--output", $VerifyResultPath)
        if ($ExpectedSnapshotId) {
            $VerifyArguments += @("--expected-snapshot-id", $ExpectedSnapshotId)
        }
        $VerifyOut = & $Python @VerifyArguments 2>&1
        $VerifyExit = if ($LASTEXITCODE -ne $null) { [int]$LASTEXITCODE } else { 1 }
        $VerifyOut | Out-File -FilePath $VerifyLog -Encoding UTF8
        $VerifyDetail = ($VerifyOut | Where-Object { "$_" -match '^VERIFY ' } | Select-Object -Last 1)
    } catch {
        $VerifyExit = 1
        $VerifyDetail = "VERIFY result=ERROR"
    } finally {
        $ErrorActionPreference = $PrevEap2
        $env:PYTHONIOENCODING = $PrevPyEnc2
    }
    $VerifyStatus = switch ($VerifyExit) {
        0 { "VERIFIED" }
        2 { "STALE" }
        3 { "FALLBACK" }
        4 { "MISMATCH" }
        5 { "REMOTE_DEGRADED" }
        6 { "LEGACY" }
        default { "ERROR" }
    }
    try {
        $VerifyData = Read-JsonFile -Path $VerifyResultPath
    } catch {
        $VerifyStatus = "ERROR"
        Write-Log "배포 검수 결과 JSON을 읽지 못함"
    }
    Write-Log "deployed_verify 종료: status=$VerifyStatus detail=$VerifyDetail"
} elseif (-not $Apply) {
    $VerifyStatus = "SKIPPED_DRY_RUN"
}

$ProductionFailed = $false
if ($Apply -and $FailedSteps.Count -eq 0) {
    if ($PublishStatus -ne "PASS") {
        $FinalStatus = "FAIL_GITHUB_PUBLISH"
        $ProductionFailed = $true
    } elseif ($VerifyStatus -ne "VERIFIED") {
        $FinalStatus = "FAIL_DEPLOYED_VERIFY"
        $ProductionFailed = $true
    }
}

# 앞에서 먼저 만든 실행 보고에 최종 운영 발행 결과를 반영해 다시 저장한다.
$Metrics.final_status = $FinalStatus
$Metrics | Add-Member -NotePropertyName git_data_push_status -NotePropertyValue $PushStatus -Force
$Metrics | Add-Member -NotePropertyName github_publish_status -NotePropertyValue $PublishStatus -Force
$Metrics | Add-Member -NotePropertyName deployed_verify_status -NotePropertyValue $VerifyStatus -Force
$Metrics | Add-Member -NotePropertyName deployed_verify_detail -NotePropertyValue $VerifyDetail -Force
$Metrics | Add-Member -NotePropertyName serving_verification -NotePropertyValue $VerifyData -Force
$Metrics | ConvertTo-Json -Depth 20 | Set-Content -Path $MetricsPath -Encoding UTF8
Copy-Item -LiteralPath $MetricsPath -Destination $LatestMetricsPath -Force

$ReportLines += @(
    "",
    "## 운영 발행",
    "",
    "- Git 데이터 push: ``$PushStatus``",
    "- GitHub 릴리스 발행 및 Render 동기화: ``$PublishStatus``",
    "- 배포 데이터 검수: ``$VerifyStatus``",
    "- 검수 상세: ``$VerifyDetail``"
)
$ReportLines[2] = "- final_status: ``$FinalStatus``"
$ReportLines -join "`n" | Set-Content -Path $ReportPath -Encoding UTF8
Copy-Item -LiteralPath $ReportPath -Destination $LatestReportPath -Force

$ScheduleLines = @(
    "# CP212 schedule status",
    "",
    "- updated_at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss KST')",
    "- runner: scripts/run_v1_unified_refresh_local.ps1",
    "- mode: $ModeText",
    "- status: $FinalStatus",
    "- reload_status: $ReloadStatus",
    "- git_data_push_status: $PushStatus",
    "- github_publish_status: $PublishStatus",
    "- deployed_verify_status: $VerifyStatus",
    "- metrics: $MetricsPath",
    "- report: $ReportPath"
)
$ScheduleLines -join "`n" | Set-Content -Path $LatestSchedulePath -Encoding UTF8

Write-Log "CP212 unified v1 refresh 종료: status=$FinalStatus reload=$ReloadStatus push=$PushStatus publish=$PublishStatus verify=$VerifyStatus"

# 정상 폴백 또는 이전 원격 버전 유지 응답은 경고색으로 알리고 실패 종료코드는 유지한다.
$SlackColor = Get-ServingNotificationColor `
    -VerifyStatus $VerifyStatus `
    -StepFailed ($FailedSteps.Count -gt 0) `
    -ProductionFailed $ProductionFailed `
    -OtherWarning (($ReloadStatus -like "WARN*") -or ($FinalStatus -like "*PARTIAL*"))
$SlackSteps = ($Steps | ForEach-Object { "$($_.name)=$($_.status)" }) -join "  "
$ServingText = "서빙 상태: 확인 못함"
if ($VerifyData) {
    $ServingText = "서빙 출처=$($VerifyData.source)  버전=$($VerifyData.snapshot_id)  폴백=$($VerifyData.fallback_active)`n원격 동기화 상태=$($VerifyData.remote_sync_status)  마지막 성공=$($VerifyData.last_remote_success_at)`n$($VerifyData.reason)"
}
$SlackText = "Lens 일일 갱신 - $RunDate`n모드=$ModeText  결과=$FinalStatus`nGitHub 릴리스 발행=$PublishStatus  배포 검수=$VerifyStatus`n$ServingText`n$VerifyDetail`n$SlackSteps"
$SlackNotify = Send-SlackNotification -Text $SlackText -Color $SlackColor
Write-Log "slack_notify: $SlackNotify"

# 실제 전송 결과는 알림 호출 뒤에야 알 수 있으므로 최종 보고서에 다시 기록한다.
$Metrics | Add-Member -NotePropertyName slack_notification_status -NotePropertyValue $SlackNotify -Force
$Metrics | Add-Member -NotePropertyName slack_notification_color -NotePropertyValue $SlackColor -Force
$Metrics | ConvertTo-Json -Depth 20 | Set-Content -Path $MetricsPath -Encoding UTF8
Copy-Item -LiteralPath $MetricsPath -Destination $LatestMetricsPath -Force
$ReportLines += @("", "## Slack 알림", "", "- 전송 상태: ``$SlackNotify``", "- 표시 색상: ``$SlackColor``")
$ReportLines -join "`n" | Set-Content -Path $ReportPath -Encoding UTF8
Copy-Item -LiteralPath $ReportPath -Destination $LatestReportPath -Force
$ScheduleLines += "- slack_notification_status: $SlackNotify"
$ScheduleLines -join "`n" | Set-Content -Path $LatestSchedulePath -Encoding UTF8

Write-Host "status=$FinalStatus"
Write-Host "push=$PushStatus"
Write-Host "publish=$PublishStatus"
Write-Host "verify=$VerifyStatus"
Write-Host "slack=$SlackNotify"
Write-Host "metrics=$MetricsPath"
Write-Host "report=$ReportPath"
Write-Host "schedule_status=$LatestSchedulePath"

if (($FailedSteps.Count -gt 0) -or $ProductionFailed) {
    exit 1
}
exit 0


