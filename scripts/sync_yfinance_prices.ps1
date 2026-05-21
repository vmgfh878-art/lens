param(
    [string[]]$Tickers = @(),
    [string]$StartDate = ((Get-Date).AddYears(-2).ToString("yyyy-MM-dd")),
    [string]$EndDate = (Get-Date).ToString("yyyy-MM-dd"),
    [string]$FallbackProvider = "eodhd",
    [string]$MetricsPath = "docs/cp86_yfinance_local_primary_migration_metrics.json",
    [switch]$Write,
    [switch]$AllowFail
)

$ErrorActionPreference = "Stop"
$env:MARKET_DATA_PROVIDER = "yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER = $FallbackProvider

$commandArgs = @(
    "-m", "backend.collector.pipelines.yfinance_price_sync",
    "--provider", "yfinance",
    "--fallback-provider", $FallbackProvider,
    "--start-date", $StartDate,
    "--end-date", $EndDate,
    "--metrics-path", $MetricsPath
)

if ($Write) {
    $commandArgs += "--write"
}
if ($AllowFail) {
    $commandArgs += "--allow-fail"
}
if ($Tickers.Count -gt 0) {
    $commandArgs += "--tickers"
    $commandArgs += $Tickers
}

python @commandArgs

