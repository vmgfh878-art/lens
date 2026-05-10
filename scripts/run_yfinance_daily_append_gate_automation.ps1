param(
    [string]$RunDate = (Get-Date).ToString("yyyy-MM-dd"),
    [int]$LookbackDays = 7,
    [string[]]$Tickers = @("AAPL", "MSFT", "NVDA", "TSLA", "NFLX"),
    [string]$RunDir = "docs\daily_yfinance_append_gate_runs",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RunDirPath = Join-Path $Root $RunDir
New-Item -ItemType Directory -Force -Path $RunDirPath | Out-Null

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SafeDate = $RunDate.Replace("-", "")
$MetricsPath = Join-Path $RunDirPath "daily_yfinance_append_gate_metrics_${SafeDate}_${RunStamp}.json"
$ReportPath = Join-Path $RunDirPath "daily_yfinance_append_gate_report_${SafeDate}_${RunStamp}.md"
$StdoutPath = Join-Path $RunDirPath "daily_yfinance_append_gate_stdout_${SafeDate}_${RunStamp}.log"
$StderrPath = Join-Path $RunDirPath "daily_yfinance_append_gate_stderr_${SafeDate}_${RunStamp}.log"
$LatestMetricsPath = Join-Path $Root "docs\daily_yfinance_append_gate_metrics.json"
$LatestReportPath = Join-Path $Root "docs\daily_yfinance_append_gate_latest.md"
$HistoryPath = Join-Path $RunDirPath "daily_yfinance_append_gate_history.csv"

$ModeText = if ($DryRun) { "DryRun" } else { "Apply" }
$DryRunText = if ($DryRun) { " -DryRun" } else { "" }
$CommandText = ".\scripts\run_yfinance_daily_append_gate_automation.ps1$DryRunText -RunDate $RunDate -LookbackDays $LookbackDays -Tickers $($Tickers -join ',')"

$ExitCode = 0
$CaughtError = $null
try {
    $env:MARKET_DATA_PROVIDER = "yfinance"
    $env:MARKET_DATA_FALLBACK_PROVIDER = ""
    $env:EODHD_API_KEY = ""
    $env:LENS_DATA_BACKEND = "local"
    $env:LENS_REQUIRE_LOCAL_SNAPSHOTS = "1"
    $env:LENS_LOCAL_SNAPSHOT_DIR = Join-Path $Root "data\parquet"
    $env:WANDB_MODE = "disabled"
    foreach ($ProxyKey in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
        $ProxyValue = [Environment]::GetEnvironmentVariable($ProxyKey)
        if ($ProxyValue -and $ProxyValue.Contains("127.0.0.1:9")) {
            Remove-Item "Env:$ProxyKey" -ErrorAction SilentlyContinue
        }
    }

    $Python = Join-Path $Root ".venv\Scripts\python.exe"
    $PythonArgs = @(
        "scripts\cp134_local_daily_update_rehearsal.py",
        "--current-date", $RunDate,
        "--lookback-days", $LookbackDays,
        "--metrics-path", $MetricsPath,
        "--tickers"
    )
    $PythonArgs += $Tickers
    if ($DryRun) {
        $PythonArgs += "--dry-run"
    } else {
        $PythonArgs += "--apply"
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
        final_decision = [pscustomobject]@{
            status = "AUTOMATION_COMMAND_FAILED"
            summary = "daily append gate metrics file was not created."
            warnings = @()
            failures = @("metrics_missing")
        }
        price_update_dry_run = [pscustomobject]@{}
        readiness = [pscustomobject]@{}
        forbidden_actions_observed = [pscustomobject]@{}
    }
}

$Decision = $Metrics.final_decision
$PriceState = $Metrics.price_update_dry_run
$Readiness = $Metrics.readiness
$Forbidden = $Metrics.forbidden_actions_observed

$Status = if ($Decision.status) { [string]$Decision.status } else { "UNKNOWN" }
$Summary = if ($Decision.summary) { [string]$Decision.summary } else { "" }
$Warnings = @()
if ($Decision.warnings) {
    $Warnings = @($Decision.warnings)
}
$Failures = @()
if ($Decision.failures) {
    $Failures = @($Decision.failures)
}

$CancelJudgement = "EODHD_CANCEL_HOLD"
if ($Status -eq "PASS_APPEND_DONE") {
    $CancelJudgement = "EODHD_CANCEL_CANDIDATE_NEEDS_2_TO_3_CONSECUTIVE_PASSES"
} elseif ($Status -eq "PARTIAL_APPEND_DONE") {
    $CancelJudgement = "EODHD_CANCEL_HOLD_PARTIAL_APPEND_NEEDS_FAILED_TICKER_RECOVERY"
} elseif ($Status -eq "PASS_WITH_NO_NEW_DAY") {
    $CancelJudgement = "NO_NEW_COMPLETED_DAY_NOT_ENOUGH_FOR_CANCEL"
}

$RunAtText = Get-Date -Format "yyyy-MM-dd HH:mm:ss KST"

$ReportLines = @(
    "# yfinance daily append gate automation result",
    "",
    "- run_at: $RunAtText",
    "- run_date: $RunDate",
    "- mode: $ModeText",
    "- exit_code: $ExitCode",
    "- status: $Status",
    "- eodhd_cancel_judgement: $CancelJudgement",
    "",
    "## Command",
    "",
    "~~~powershell",
    $CommandText,
    "~~~",
    "",
    "## Summary",
    "",
    $Summary,
    "",
    "## Snapshot",
    "",
    "- yfinance_price_latest_date: $($Readiness.local_price_latest_date)",
    "- yfinance_indicator_1d_latest_date: $($Readiness.local_indicator_1D_latest_date)",
    "- yfinance_indicator_1w_latest_date: $($Readiness.local_indicator_1W_latest_date)",
    "- product_history_latest_asof_date: $($Readiness.product_history_latest_asof_date)",
    "- fetch_status: $($PriceState.status)",
    "- append_candidate_rows: $($PriceState.completed_append_candidate_rows)",
    "- append_candidate_tickers: $(@($PriceState.append_candidate_tickers) -join ', ')",
    "- successful_fetch_tickers: $(@($PriceState.successful_fetch_tickers) -join ', ')",
    "- empty fetch tickers: $(@($PriceState.empty_fetch_tickers) -join ', ')",
    "- fetch failed tickers: $(@($PriceState.fetch_failed_tickers) -join ', ')",
    "- failed tickers: $(@($PriceState.failed_tickers) -join ', ')",
    "- fallback used count: $($PriceState.fallback_used_count)",
    "",
    "## Forbidden Action Check",
    "",
    "- eodhd_call: $($Forbidden.eodhd_call)",
    "- Supabase bulk read: $($Forbidden.supabase_bulk_read)",
    "- Supabase bulk write: $($Forbidden.supabase_price_data_indicators_context_bulk_write)",
    "- model_training: $($Forbidden.model_training)",
    "- full_inference_save: $($Forbidden.full_inference_save)",
    "",
    "## Warnings",
    ""
)
if ($Warnings.Count -eq 0) {
    $ReportLines += "- none"
} else {
    foreach ($Warning in $Warnings) {
        $ReportLines += "- $Warning"
    }
}

$ReportLines += @(
    "",
    "## Failures",
    ""
)
if ($Failures.Count -eq 0) {
    $ReportLines += "- none"
} else {
    foreach ($Failure in $Failures) {
        $ReportLines += "- $Failure"
    }
}

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

$ReportLines += @(
    "",
    "## Artifacts",
    "",
    "- metrics: $MetricsPath",
    "- stdout: $StdoutPath",
    "- stderr: $StderrPath"
)

$ReportLines -join "`n" | Set-Content -Path $ReportPath -Encoding UTF8
Copy-Item -LiteralPath $ReportPath -Destination $LatestReportPath -Force

if (-not (Test-Path -LiteralPath $HistoryPath)) {
    "run_at,run_date,mode,exit_code,status,price_latest,indicator_1d_latest,append_candidate_rows,empty_fetch_count,fetch_failed_count,fallback_used_count,cancel_judgement,report_path,metrics_path" |
        Set-Content -Path $HistoryPath -Encoding UTF8
}

$HistoryRow = @(
    ('"' + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + '"'),
    ('"' + $RunDate + '"'),
    ('"' + $ModeText + '"'),
    $ExitCode,
    ('"' + $Status + '"'),
    ('"' + $Readiness.local_price_latest_date + '"'),
    ('"' + $Readiness.local_indicator_1D_latest_date + '"'),
    ('"' + $PriceState.completed_append_candidate_rows + '"'),
    ('"' + $PriceState.empty_fetch_count + '"'),
    ('"' + @($PriceState.fetch_failed_tickers).Count + '"'),
    ('"' + $PriceState.fallback_used_count + '"'),
    ('"' + $CancelJudgement + '"'),
    ('"' + $ReportPath + '"'),
    ('"' + $MetricsPath + '"')
) -join ","
Add-Content -Path $HistoryPath -Value $HistoryRow -Encoding UTF8

Write-Host "status=$Status"
Write-Host "report=$ReportPath"
Write-Host "metrics=$MetricsPath"
Write-Host "latest_report=$LatestReportPath"

exit $ExitCode
