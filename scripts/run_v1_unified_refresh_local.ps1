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

$LineMetricsPath = Join-Path $RunDirPath "line_export_metrics_${SafeDate}_${RunStamp}.json"
$LineReportPath = Join-Path $RunDirPath "line_export_report_${SafeDate}_${RunStamp}.md"
Invoke-PythonStep -Name "line_export" -Arguments @(
    "backend\scripts\cp212_line_1d_export.py",
    $ModeArg,
    "--metrics-path", $LineMetricsPath,
    "--report-path", $LineReportPath
) -StdoutPath (Join-Path $RunDirPath "line_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "line_stderr_${SafeDate}_${RunStamp}.log") | Out-Null

if ($Apply) {
    Invoke-PythonStep -Name "product_history_rebuild" -Arguments @("backend\scripts\rebuild_product_history_parquet.py") -StdoutPath (Join-Path $RunDirPath "history_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "history_stderr_${SafeDate}_${RunStamp}.log") | Out-Null
    Invoke-PythonStep -Name "ai_runs_mock_rebuild" -Arguments @("backend\scripts\build_ai_runs_mock.py") -StdoutPath (Join-Path $RunDirPath "ai_runs_stdout_${SafeDate}_${RunStamp}.log") -StderrPath (Join-Path $RunDirPath "ai_runs_stderr_${SafeDate}_${RunStamp}.log") | Out-Null
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
    "- line은 CP212 F4 beta=4 ensemble artifact를 serving parquet로 export"
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

Write-Log "CP212 unified v1 refresh 종료: status=$FinalStatus reload=$ReloadStatus"
Write-Host "status=$FinalStatus"
Write-Host "metrics=$MetricsPath"
Write-Host "report=$ReportPath"
Write-Host "schedule_status=$LatestSchedulePath"

if ($FailedSteps.Count -gt 0) {
    exit 1
}
exit 0


