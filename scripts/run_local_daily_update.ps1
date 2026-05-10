param(
    [string[]]$Tickers = @("AAPL", "MSFT", "NVDA", "TSLA", "NFLX"),
    [string]$CurrentDate = (Get-Date).ToString("yyyy-MM-dd"),
    [int]$LookbackDays = 10,
    [string]$MetricsPath = "docs/cp134_local_daily_update_pipeline_metrics.json",
    [switch]$DryRun,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$env:MARKET_DATA_PROVIDER = "yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER = ""
$env:EODHD_API_KEY = ""
$env:LENS_DATA_BACKEND = "local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS = "1"
$env:LENS_LOCAL_SNAPSHOT_DIR = "C:\Users\user\lens\data\parquet"
$env:WANDB_MODE = "disabled"

# 로컬 차단 프록시가 yfinance 호출을 빈 응답으로 오인하게 만들지 않도록 제거한다.
foreach ($proxyKey in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
    $proxyValue = [Environment]::GetEnvironmentVariable($proxyKey)
    if ($proxyValue -and $proxyValue.Contains("127.0.0.1:9")) {
        Remove-Item "Env:$proxyKey" -ErrorAction SilentlyContinue
    }
}

if ($Apply -and $DryRun) {
    throw "-Apply와 -DryRun은 동시에 사용할 수 없습니다."
}

$argsList = @(
    "scripts\cp134_local_daily_update_rehearsal.py",
    "--current-date", $CurrentDate,
    "--lookback-days", $LookbackDays,
    "--metrics-path", $MetricsPath,
    "--tickers"
)
$argsList += $Tickers

if ($Apply) {
    Write-Host "명시적 -Apply가 감지되어 yfinance local parquet actual append gate를 엽니다."
    $argsList += "--apply"
} else {
    Write-Host "기본값은 dry-run입니다. 실제 append는 .\scripts\run_local_daily_update.ps1 -Apply 로만 실행됩니다."
    $argsList += "--dry-run"
}

.\.venv\Scripts\python.exe @argsList
