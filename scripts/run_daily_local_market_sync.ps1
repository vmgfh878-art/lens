param(
    [string]$Universe = "backend/data/universe/sp500.csv",
    [string]$StartDate = ((Get-Date).AddDays(-10).ToString("yyyy-MM-dd")),
    [string]$EndDate = (Get-Date).ToString("yyyy-MM-dd"),
    [int]$PriceBatchLimit = 80,
    [int]$IndicatorLookbackDays = 60,
    [string]$FallbackProvider = "eodhd",
    [string]$MetricsPath = "docs/cp86_yfinance_local_primary_migration_metrics.json",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:MARKET_DATA_PROVIDER = "yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER = $FallbackProvider

$priceArgs = @(
    "-m", "backend.collector.pipelines.yfinance_price_sync",
    "--provider", "yfinance",
    "--fallback-provider", $FallbackProvider,
    "--start-date", $StartDate,
    "--end-date", $EndDate,
    "--batch-limit", $PriceBatchLimit,
    "--metrics-path", $MetricsPath
)

if (-not $DryRun) {
    $priceArgs += "--write"
    $priceArgs += "--universe"
    $priceArgs += $Universe
}

python @priceArgs

if ($DryRun) {
    Write-Host "DRY-RUN: indicator compute와 data quality check는 실행하지 않고 순서만 확인했습니다."
    Write-Host "DRY-RUN NEXT: python -m backend.collector.pipelines.compute_indicators_cli --lookback-days $IndicatorLookbackDays --timeframes 1D 1W 1M"
    Write-Host "DRY-RUN NEXT: python -m backend.collector.pipelines.data_coverage_report"
    Write-Host "DRY-RUN NEXT: live inference는 CP87에서 실행하지 않습니다."
    exit 0
}

python -m backend.collector.pipelines.compute_indicators_cli `
    --lookback-days $IndicatorLookbackDays `
    --timeframes 1D 1W 1M

python -m backend.collector.pipelines.data_coverage_report
