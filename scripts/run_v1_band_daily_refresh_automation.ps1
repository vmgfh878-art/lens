param(
    [string]$RunDate = (Get-Date).ToString("yyyy-MM-dd"),
    [switch]$SkipAppend,
    [switch]$DryRun,
    [switch]$Apply,
    [switch]$ReloadLocalBackend,
    [string]$AdminReloadUrl = "http://127.0.0.1:8000/api/v1/admin/reload",
    [string]$RunDir = "docs\cp210_band_refresh_runs"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($DryRun -and $Apply) {
    throw "-DryRun과 -Apply는 동시에 사용할 수 없습니다."
}
if (-not $DryRun -and -not $Apply) {
    throw "운영 자동화는 -Apply 또는 -DryRun 중 하나를 명시해야 합니다."
}

$RunDirPath = Join-Path $Root $RunDir
$LogDir = Join-Path $Root "logs\cp210_band_refresh"
New-Item -ItemType Directory -Force -Path $RunDirPath | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SafeDate = $RunDate.Replace("-", "")
$AppendMetricsPath = Join-Path $RunDirPath "append_metrics_${SafeDate}_${RunStamp}.json"
$AppendReportPath = Join-Path $RunDirPath "append_report_${SafeDate}_${RunStamp}.md"
$AppendStdoutPath = Join-Path $RunDirPath "append_stdout_${SafeDate}_${RunStamp}.log"
$AppendStderrPath = Join-Path $RunDirPath "append_stderr_${SafeDate}_${RunStamp}.log"
$BandMetricsPath = Join-Path $RunDirPath "band_refresh_metrics_${SafeDate}_${RunStamp}.json"
$BandReportPath = Join-Path $RunDirPath "band_refresh_report_${SafeDate}_${RunStamp}.md"
$BandStdoutPath = Join-Path $RunDirPath "band_refresh_stdout_${SafeDate}_${RunStamp}.log"
$BandStderrPath = Join-Path $RunDirPath "band_refresh_stderr_${SafeDate}_${RunStamp}.log"
$LatestMetricsPath = Join-Path $Root "docs\cp210_band_refresh_metrics.json"
$LatestReportPath = Join-Path $Root "docs\cp210_band_refresh_report.md"
$LatestSchedulePath = Join-Path $Root "docs\cp210_band_refresh_schedule_status.md"
$HistoryPath = Join-Path $RunDirPath "cp210_band_refresh_history.csv"
$PipelineLogPath = Join-Path $LogDir "pipeline_${SafeDate}_${RunStamp}.log"

function Write-Log {
    param([string]$Message)
    $line = "[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $Message
    Write-Host $line
    Add-Content -Path $PipelineLogPath -Value $line -Encoding UTF8
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

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
$ModeArg = if ($Apply) { "--apply" } else { "--dry-run" }
$ModeText = if ($Apply) { "Apply" } else { "DryRun" }
$AppendExitCode = 0
$BandExitCode = 0
$ReloadStatus = "SKIPPED"

Write-Log "CP210 band daily refresh 시작: mode=$ModeText run_date=$RunDate skip_append=$SkipAppend"

if (-not $SkipAppend) {
    Write-Log "yfinance 500 append 시작"
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
    if ($Apply) {
        $AppendArgs += "--apply"
    } else {
        $AppendArgs += "--dry-run"
    }
    & $Python @AppendArgs 1> $AppendStdoutPath 2> $AppendStderrPath
    $AppendExitCode = if ($LASTEXITCODE -ne $null) { [int]$LASTEXITCODE } else { 0 }
    Write-Log "yfinance 500 append 종료: exit_code=$AppendExitCode"
    if ($AppendExitCode -ne 0) {
        throw "append 단계가 실패해서 band refresh를 중단합니다. stderr=$AppendStderrPath"
    }
} else {
    Write-Log "append 단계 skip"
}

Write-Log "band forward refresh 시작"
$BandArgs = @(
    "backend\scripts\cp210_band_forward_refresh.py",
    $ModeArg,
    "--batch-size", "2048",
    "--metrics-path", $BandMetricsPath,
    "--report-path", $BandReportPath
)
& $Python @BandArgs 1> $BandStdoutPath 2> $BandStderrPath
$BandExitCode = if ($LASTEXITCODE -ne $null) { [int]$LASTEXITCODE } else { 0 }
Write-Log "band forward refresh 종료: exit_code=$BandExitCode"
if ($BandExitCode -ne 0) {
    throw "band refresh 단계가 실패했습니다. stderr=$BandStderrPath"
}

Copy-Item -LiteralPath $BandMetricsPath -Destination $LatestMetricsPath -Force
Copy-Item -LiteralPath $BandReportPath -Destination $LatestReportPath -Force

$BandMetrics = Read-JsonFile -Path $BandMetricsPath

if ($ReloadLocalBackend) {
    $Token = [Environment]::GetEnvironmentVariable("LENS_ADMIN_RELOAD_TOKEN")
    if (-not $Token) {
        $ReloadStatus = "SKIPPED_TOKEN_MISSING"
        Write-Log "admin reload token이 없어 local backend reload를 건너뜁니다."
    } else {
        try {
            Invoke-RestMethod -Method Post -Uri $AdminReloadUrl -Headers @{ "X-Lens-Admin-Token" = $Token } | Out-Null
            $ReloadStatus = "PASS"
            Write-Log "local backend admin reload 완료"
        } catch {
            $ReloadStatus = "WARN_RELOAD_FAILED"
            Write-Log "local backend admin reload 실패: $($_.Exception.Message)"
        }
    }
}

if (-not (Test-Path -LiteralPath $HistoryPath)) {
    "run_at,run_date,mode,append_exit_code,band_exit_code,status,price_latest,indicator_1d_latest,indicator_1w_latest,band_1d_asof,band_1w_asof,reload_status,metrics_path,report_path" |
        Set-Content -Path $HistoryPath -Encoding UTF8
}

$Status = if ($BandMetrics.final_status) { [string]$BandMetrics.final_status } else { "UNKNOWN" }
$HistoryRow = @(
    ('"' + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + '"'),
    ('"' + $RunDate + '"'),
    ('"' + $ModeText + '"'),
    $AppendExitCode,
    $BandExitCode,
    ('"' + $Status + '"'),
    ('"' + $BandMetrics.inputs.price.date_max + '"'),
    ('"' + $BandMetrics.inputs.indicators_1d.date_max + '"'),
    ('"' + $BandMetrics.inputs.indicators_1w.date_max + '"'),
    ('"' + $BandMetrics.after.band_1d.asof_max + '"'),
    ('"' + $BandMetrics.after.band_1w.asof_max + '"'),
    ('"' + $ReloadStatus + '"'),
    ('"' + $BandMetricsPath + '"'),
    ('"' + $BandReportPath + '"')
) -join ","
Add-Content -Path $HistoryPath -Value $HistoryRow -Encoding UTF8

$ScheduleLines = @(
    "# CP210 band refresh schedule status",
    "",
    "- updated_at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss KST')",
    "- runner: scripts/run_v1_band_daily_refresh_automation.ps1",
    "- mode: $ModeText",
    "- append_exit_code: $AppendExitCode",
    "- band_exit_code: $BandExitCode",
    "- status: $Status",
    "- reload_status: $ReloadStatus",
    "- price_latest: $($BandMetrics.inputs.price.date_max)",
    "- indicator_1d_latest: $($BandMetrics.inputs.indicators_1d.date_max)",
    "- indicator_1w_latest: $($BandMetrics.inputs.indicators_1w.date_max)",
    "- band_1d_asof: $($BandMetrics.after.band_1d.asof_max)",
    "- band_1w_asof: $($BandMetrics.after.band_1w.asof_max)",
    "- metrics: $BandMetricsPath",
    "- report: $BandReportPath"
)
$ScheduleLines -join "`n" | Set-Content -Path $LatestSchedulePath -Encoding UTF8

Write-Log "CP210 band daily refresh 완료: status=$Status reload=$ReloadStatus"
Write-Host "status=$Status"
Write-Host "metrics=$BandMetricsPath"
Write-Host "report=$BandReportPath"
Write-Host "schedule_status=$LatestSchedulePath"

exit $BandExitCode
