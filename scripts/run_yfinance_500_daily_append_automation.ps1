param(
    [string]$RunDate = (Get-Date).ToString("yyyy-MM-dd"),
    [string]$StartDate = "2015-01-01",
    [int]$ChunkSize = 75,
    [int]$MaxChunksPerRun = 8,
    [double]$SleepSecondsBetweenTickers = 0.2,
    [double]$SleepSecondsBetweenChunks = 30,
    [int]$IncrementalLookbackDays = 10,
    [string]$RunDir = "docs\yfinance_500_daily_append_runs",
    [switch]$DryRun,
    [switch]$Apply
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
New-Item -ItemType Directory -Force -Path $RunDirPath | Out-Null

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SafeDate = $RunDate.Replace("-", "")
$MetricsPath = Join-Path $RunDirPath "yfinance_500_daily_append_metrics_${SafeDate}_${RunStamp}.json"
$ReportPath = Join-Path $RunDirPath "yfinance_500_daily_append_report_${SafeDate}_${RunStamp}.md"
$StdoutPath = Join-Path $RunDirPath "yfinance_500_daily_append_stdout_${SafeDate}_${RunStamp}.log"
$StderrPath = Join-Path $RunDirPath "yfinance_500_daily_append_stderr_${SafeDate}_${RunStamp}.log"
$FailedCsvPath = Join-Path $RunDirPath "yfinance_500_daily_append_failed_tickers_${SafeDate}_${RunStamp}.csv"
$DistributionCsvPath = Join-Path $RunDirPath "yfinance_500_daily_append_latest_distribution_${SafeDate}_${RunStamp}.csv"
$LatestMetricsPath = Join-Path $Root "docs\yfinance_500_daily_append_metrics.json"
$LatestReportPath = Join-Path $Root "docs\yfinance_500_daily_append_latest.md"
$HistoryPath = Join-Path $RunDirPath "yfinance_500_daily_append_history.csv"

$ModeText = if ($Apply) { "Apply" } else { "DryRun" }
$ExitCode = 0
$CaughtError = $null
$PreviousErrorActionPreference = $null

try {
    # 이 자동화는 500티커 local parquet 전용이다. EODHD, Supabase, 모델 저장 경로를 쓰지 않는다.
    $env:MARKET_DATA_PROVIDER = "yfinance"
    $env:MARKET_DATA_FALLBACK_PROVIDER = ""
    $env:EODHD_API_KEY = ""
    $env:LENS_DATA_BACKEND = "local"
    $env:LENS_REQUIRE_LOCAL_SNAPSHOTS = "1"
    $env:LENS_LOCAL_SNAPSHOT_DIR = Join-Path $Root "data\parquet"
    $env:WANDB_MODE = "disabled"
    $env:YFINANCE_FETCH_METHOD = "direct_chart"
    $env:PYTHONUTF8 = "1"
    foreach ($ProxyKey in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
        $ProxyValue = [Environment]::GetEnvironmentVariable($ProxyKey)
        if ($ProxyValue -and $ProxyValue.Contains("127.0.0.1:9")) {
            Remove-Item "Env:$ProxyKey" -ErrorAction SilentlyContinue
        }
    }

    $Python = Join-Path $Root ".venv\Scripts\python.exe"
    $PythonArgs = @(
        "scripts\cp151_yfinance_500_backfill.py",
        "--start-date", $StartDate,
        "--end-date", $RunDate,
        "--chunk-size", $ChunkSize,
        "--max-chunks-per-run", $MaxChunksPerRun,
        "--sleep-seconds-between-tickers", $SleepSecondsBetweenTickers,
        "--sleep-seconds-between-chunks", $SleepSecondsBetweenChunks,
        "--fetch-mode", "incremental",
        "--incremental-lookback-days", $IncrementalLookbackDays,
        "--indicator-mode", "auto",
        "--metrics-path", $MetricsPath,
        "--report-path", $ReportPath,
        "--failed-tickers-csv", $FailedCsvPath,
        "--latest-distribution-csv", $DistributionCsvPath
    )
    if ($Apply) {
        $PythonArgs += "--apply"
    } else {
        $PythonArgs += "--dry-run"
    }

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @PythonArgs 1> $StdoutPath 2> $StderrPath
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        $ExitCode = [int]$LASTEXITCODE
    }
} catch {
    if ($PreviousErrorActionPreference) {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    $ExitCode = 1
    $CaughtError = $_
    $CaughtError | Out-String | Set-Content -Path $StderrPath -Encoding UTF8
}

$Metrics = $null
if (Test-Path -LiteralPath $MetricsPath) {
    try {
        $Metrics = Get-Content -LiteralPath $MetricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
        Copy-Item -LiteralPath $MetricsPath -Destination $LatestMetricsPath -Force
    } catch {
        $ExitCode = 1
        $CaughtError = $_
        $CaughtError | Out-String | Add-Content -Path $StderrPath -Encoding UTF8
    }
}

if ($null -eq $Metrics) {
    $Metrics = [pscustomobject]@{
        final_status = "AUTOMATION_COMMAND_FAILED"
        final_summary = "yfinance 500 daily append metrics file was not created."
        price_backfill = [pscustomobject]@{}
        latest_distribution = [pscustomobject]@{}
        forbidden_actions_observed = [pscustomobject]@{}
    }
}

$Status = if ($Metrics.final_status) { [string]$Metrics.final_status } else { "UNKNOWN" }
$Summary = if ($Metrics.final_summary) { [string]$Metrics.final_summary } else { "" }
$PriceState = $Metrics.price_backfill
$Distribution = $Metrics.latest_distribution
$Forbidden = $Metrics.forbidden_actions_observed
$RunAtText = Get-Date -Format "yyyy-MM-dd HH:mm:ss KST"

$ReportLines = @(
    "# yfinance 500 daily append automation result",
    "",
    "- run_at: $RunAtText",
    "- run_date: $RunDate",
    "- mode: $ModeText",
    "- exit_code: $ExitCode",
    "- status: $Status",
    "- fetch_mode: incremental",
    "- incremental_lookback_days: $IncrementalLookbackDays",
    "",
    "## Summary",
    "",
    $Summary,
    "",
    "## Metrics",
    "",
    "- selected_ticker_count: $($PriceState.selected_ticker_count)",
    "- success_ticker_count: $($PriceState.success_ticker_count)",
    "- failed_ticker_count: $($PriceState.failed_ticker_count)",
    "- appended_rows: $($PriceState.merge.appended_rows)",
    "- skipped_existing_rows: $($PriceState.merge.skipped_existing_rows)",
    "- fallback_used_count: $($PriceState.fallback_used_count)",
    "- complete_ratio: $($Distribution.complete_ratio)",
    "- date_max: $($Distribution.date_max)",
    "",
    "## Forbidden Action Check",
    "",
    "- eodhd_fallback: $($Forbidden.eodhd_fallback)",
    "- supabase_bulk_read_write: $($Forbidden.supabase_bulk_read_write)",
    "- db_write: $($Forbidden.db_write)",
    "- model_training: $($Forbidden.model_training)",
    "- inference_save: $($Forbidden.inference_save)",
    "",
    "## Artifacts",
    "",
    "- metrics: $MetricsPath",
    "- report: $ReportPath",
    "- failed_tickers: $FailedCsvPath",
    "- latest_distribution: $DistributionCsvPath",
    "- stdout: $StdoutPath",
    "- stderr: $StderrPath"
)

if ($CaughtError) {
    $ReportLines += @(
        "",
        "## Automation Error",
        "",
        "~~~text",
        ($CaughtError | Out-String).Trim(),
        "~~~"
    )
}

$ReportLines -join "`n" | Set-Content -Path $ReportPath -Encoding UTF8
Copy-Item -LiteralPath $ReportPath -Destination $LatestReportPath -Force

if (-not (Test-Path -LiteralPath $HistoryPath)) {
    "run_at,run_date,mode,exit_code,status,selected_ticker_count,success_ticker_count,failed_ticker_count,appended_rows,complete_ratio,date_max,report_path,metrics_path" |
        Set-Content -Path $HistoryPath -Encoding UTF8
}

$HistoryRow = @(
    ('"' + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + '"'),
    ('"' + $RunDate + '"'),
    ('"' + $ModeText + '"'),
    $ExitCode,
    ('"' + $Status + '"'),
    ('"' + $PriceState.selected_ticker_count + '"'),
    ('"' + $PriceState.success_ticker_count + '"'),
    ('"' + $PriceState.failed_ticker_count + '"'),
    ('"' + $PriceState.merge.appended_rows + '"'),
    ('"' + $Distribution.complete_ratio + '"'),
    ('"' + $Distribution.date_max + '"'),
    ('"' + $ReportPath + '"'),
    ('"' + $MetricsPath + '"')
) -join ","
Add-Content -Path $HistoryPath -Value $HistoryRow -Encoding UTF8

Write-Host "status=$Status"
Write-Host "report=$ReportPath"
Write-Host "metrics=$MetricsPath"
Write-Host "latest_report=$LatestReportPath"

exit $ExitCode
