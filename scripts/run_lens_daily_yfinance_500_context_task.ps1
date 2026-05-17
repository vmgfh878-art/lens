param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PipelineScript = Join-Path $Root "scripts\cp189_daily_yfinance_500_context_pipeline.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python 실행 파일을 찾을 수 없습니다: $Python"
}
if (-not (Test-Path -LiteralPath $PipelineScript)) {
    throw "일일 데이터 파이프라인 스크립트를 찾을 수 없습니다: $PipelineScript"
}

$ModeArg = if ($DryRun) { "--dry-run" } else { "--apply" }

# Windows 작업 스케줄러의 명령 길이 제한을 피하기 위한 짧은 진입점이다.
& $Python $PipelineScript `
    $ModeArg `
    --chunk-size 75 `
    --max-chunks-per-run 8 `
    --sleep-seconds-between-tickers 0.2 `
    --sleep-seconds-between-chunks 30 `
    --incremental-lookback-days 10

exit $LASTEXITCODE
