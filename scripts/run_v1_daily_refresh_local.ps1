param(
    [ValidateSet("local", "render")]
    [string]$Mode = "local",
    [switch]$SkipAppend,
    [switch]$PreflightOnly,
    [switch]$AllowStaticFallbackRebuild
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs\cp206_daily_refresh"
$LogPath = Join-Path $LogDir ("refresh_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Step {
    param([string]$Message)
    $line = "[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $Message
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

Set-Location $Root
Write-Step "CP206 local refresh 시작: mode=$Mode skip_append=$SkipAppend preflight_only=$PreflightOnly"

if (-not $SkipAppend) {
    Write-Step "yfinance 500 incremental append 시작"
    python scripts\cp151_yfinance_500_backfill.py --apply --fetch-mode incremental --indicator-mode incremental --metrics-path docs\cp206_append_metrics.json --report-path docs\cp206_append_report.md --failed-tickers-csv docs\cp206_append_failed_tickers.csv --latest-distribution-csv docs\cp206_append_latest_distribution.csv
    Write-Step "yfinance 500 incremental append 완료"
} else {
    Write-Step "append 단계 skip"
}

Write-Step "CP206 Stage 0/0.5 preflight 시작"
python backend\scripts\cp206_v1_local_refresh.py --preflight-only
Write-Step "CP206 Stage 0/0.5 preflight 완료"

if ($PreflightOnly) {
    Write-Step "preflight-only 요청으로 여기서 종료"
    exit 0
}

if (-not $AllowStaticFallbackRebuild) {
    Write-Step "CP206 원칙상 정적 cp204 payload rebuild는 자동 갱신 본체가 아니므로 여기서 중단"
    Write-Step "full frozen inference refresh 구현/통과 후 serving parquet rebuild를 실행해야 함"
    exit 0
}

Write-Step "명시적 허용에 따라 정적 fallback serving parquet rebuild 시작"
python backend\scripts\build_v1_predictions_local.py
python backend\scripts\rebuild_product_history_parquet.py
Write-Step "정적 fallback serving parquet rebuild 완료"

if ($Mode -eq "local") {
    Write-Step "local mode: 실행 중인 서버가 있으면 POST /api/v1/admin/reload 호출로 cache reload 가능"
} else {
    Write-Step "render mode: 산출물 commit/push 후 redeploy 기준으로 cache reload"
}

Write-Step "CP206 local refresh 종료"
