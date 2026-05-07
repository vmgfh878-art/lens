param(
    [string[]]$Tickers = @("AAPL", "MSFT", "NVDA", "TSLA", "NFLX"),
    [string]$CurrentDate = (Get-Date).ToString("yyyy-MM-dd"),
    [int]$LookbackDays = 10,
    [string]$MetricsPath = "docs/cp134_local_daily_update_pipeline_metrics.json",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$env:MARKET_DATA_PROVIDER = "yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER = ""
$env:EODHD_API_KEY = ""
$env:LENS_DATA_BACKEND = "local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS = "1"
$env:LENS_LOCAL_SNAPSHOT_DIR = "C:\Users\user\lens\data\parquet"
$env:WANDB_MODE = "disabled"

# yfinance 호출이 로컬 차단용 프록시로 빠지지 않도록 daily wrapper에서도 정리한다.
foreach ($proxyKey in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
    $proxyValue = [Environment]::GetEnvironmentVariable($proxyKey)
    if ($proxyValue -and $proxyValue.Contains("127.0.0.1:9")) {
        Remove-Item "Env:$proxyKey" -ErrorAction SilentlyContinue
    }
}

if (-not $DryRun) {
    Write-Host "CP134 기준으로 이 wrapper는 아직 dry-run 전용입니다. 실제 append/upload는 별도 승인 CP에서만 실행하세요."
    Write-Host "dry-run 실행으로 전환합니다."
}

$argsList = @(
    "scripts\cp134_local_daily_update_rehearsal.py",
    "--current-date", $CurrentDate,
    "--lookback-days", $LookbackDays,
    "--metrics-path", $MetricsPath,
    "--tickers"
)
$argsList += $Tickers

.\.venv\Scripts\python.exe @argsList
