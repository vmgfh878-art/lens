# Lens 데모용 백엔드/프론트 개발 서버를 별도 PowerShell 창에서 실행한다.

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$LogDir = Join-Path $RootDir "logs"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$BackendRunner = Join-Path $LogDir "run_backend_demo.ps1"
$FrontendRunner = Join-Path $LogDir "run_frontend_demo.ps1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

@"
`$ErrorActionPreference = "Stop"
`$Host.UI.RawUI.WindowTitle = "Lens backend 8000"
Set-Location "$RootDir"
`$env:PYTHONPATH="$BackendDir"
`$env:BACKEND_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
& "$PythonExe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"@ | Set-Content -Path $BackendRunner -Encoding UTF8

@"
`$ErrorActionPreference = "Stop"
`$Host.UI.RawUI.WindowTitle = "Lens frontend 3000"
Set-Location "$FrontendDir"
`$env:NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:8000"
npm run dev -- --hostname 127.0.0.1 --port 3000
"@ | Set-Content -Path $FrontendRunner -Encoding UTF8

Start-Process -FilePath "powershell.exe" -ArgumentList @(
  "-NoExit",
  "-ExecutionPolicy",
  "Bypass",
  "-File",
  $BackendRunner
)

Start-Sleep -Seconds 2

Start-Process -FilePath "powershell.exe" -ArgumentList @(
  "-NoExit",
  "-ExecutionPolicy",
  "Bypass",
  "-File",
  $FrontendRunner
)

Write-Host "Lens 데모 서버 실행을 요청했습니다."
Write-Host "백엔드:  http://127.0.0.1:8000"
Write-Host "프론트:  http://127.0.0.1:3000"
Write-Host "상태 확인: powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1"
