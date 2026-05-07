# Lens 데모 백엔드와 프론트 개발 서버를 실행한다.
# npm run build 이후에는 기존 next dev 서버의 static asset 경로가 어긋날 수 있으므로
# 프론트는 항상 재시작 스크립트를 거쳐 실행한다.

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$LogDir = Join-Path $RootDir "logs"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$BackendRunner = Join-Path $LogDir "run_backend_demo.ps1"
$RestartFrontend = Join-Path $PSScriptRoot "restart_frontend_dev.ps1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

@"
`$ErrorActionPreference = "Stop"
`$Host.UI.RawUI.WindowTitle = "Lens backend 8000"
Set-Location "$RootDir"
`$env:PYTHONPATH="$BackendDir"
`$env:BACKEND_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
`$env:MARKET_DATA_PROVIDER="yfinance"
`$env:LENS_USE_LOCAL_SNAPSHOTS="1"
`$env:LENS_LOCAL_SNAPSHOT_DIR="$RootDir\data\parquet"
& "$PythonExe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"@ | Set-Content -Path $BackendRunner -Encoding UTF8

$backendListening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $backendListening) {
  Start-Process -WindowStyle Hidden -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile",
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $BackendRunner
  )
  Start-Sleep -Seconds 2
} else {
  Write-Host "백엔드 8000 포트는 이미 실행 중입니다."
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RestartFrontend

Write-Host "Lens 데모 서버 실행 요청이 완료되었습니다."
Write-Host "백엔드:  http://127.0.0.1:8000"
Write-Host "프론트:  http://127.0.0.1:3000"
Write-Host "프론트 로그: logs/frontend_dev.out.log, logs/frontend_dev.err.log"
Write-Host "상태 확인: powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1"
Write-Host "주의: npm run build 후에는 프론트 dev 서버를 재시작해야 static asset 404를 피할 수 있습니다."
