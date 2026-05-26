# Lens 데모 백엔드와 프론트 개발 서버를 실행한다.
# npm run build 이후에는 기존 next dev 서버의 static asset 경로가 어긋날 수 있으므로
# 프론트는 항상 재시작 스크립트를 거쳐 실행한다.

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$LogDir = Join-Path $RootDir "logs"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$BackendOutLog = Join-Path $LogDir "backend_dev.out.log"
$BackendErrLog = Join-Path $LogDir "backend_dev.err.log"
$FrontendDir = Join-Path $RootDir "frontend"
$FrontendOutLog = Join-Path $LogDir "frontend_dev.out.log"
$FrontendErrLog = Join-Path $LogDir "frontend_dev.err.log"
$FrontendUrl = "http://127.0.0.1:3000"
$BackendUrl = "http://127.0.0.1:8000"
$FrontendHost = "127.0.0.1"
$FrontendPort = 3000
$FrontendTimeoutSeconds = 60

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Test-BackendReady {
  try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health/live" -UseBasicParsing -TimeoutSec 3
    return ([int]$response.StatusCode -eq 200)
  } catch {
    return $false
  }
}

function Start-BackendDev {
  "" | Set-Content -Path $BackendOutLog -Encoding UTF8
  "" | Set-Content -Path $BackendErrLog -Encoding UTF8

  $command = "cd /d `"$RootDir`" && " +
    "set PYTHONPATH=$BackendDir && " +
    "set BACKEND_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000 && " +
    "set MARKET_DATA_PROVIDER=yfinance && " +
    "set LENS_USE_LOCAL_SNAPSHOTS=1 && " +
    "set LENS_LOCAL_SNAPSHOT_DIR=$RootDir\data\parquet && " +
    "`"$PythonExe`" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 1>> `"$BackendOutLog`" 2>> `"$BackendErrLog`""

  $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $processInfo.FileName = "cmd.exe"
  $processInfo.Arguments = "/d /s /c `"$command`""
  $processInfo.WorkingDirectory = $RootDir
  $processInfo.UseShellExecute = $false
  $processInfo.CreateNoWindow = $true
  $processInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden

  $process = [System.Diagnostics.Process]::Start($processInfo)
  Write-Host "백엔드 dev server launcher PID=$($process.Id)"
  Write-Host "백엔드 stdout: $BackendOutLog"
  Write-Host "백엔드 stderr: $BackendErrLog"
}

function Stop-FrontendPortOwner {
  $listeners = @(Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue)
  foreach ($listener in $listeners) {
    $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    if (-not $process) {
      continue
    }

    if ($process.ProcessName -in @("node", "cmd", "npm")) {
      Write-Host "프론트 포트 점유 프로세스 종료: PID=$($process.Id) name=$($process.ProcessName)"
      Stop-Process -Id $process.Id -Force
    } else {
      Write-Host "포트 $FrontendPort 점유 프로세스는 자동 종료하지 않습니다: $($process.ProcessName) PID=$($process.Id)"
    }
  }
}

function Resolve-FrontendUrl {
  param([string]$Path)

  if ($Path.StartsWith("http://") -or $Path.StartsWith("https://")) {
    return $Path
  }

  $root = $FrontendUrl.TrimEnd("/")
  if ($Path.StartsWith("/")) {
    return "$root$Path"
  }

  return "$root/$Path"
}

function Test-FrontendReady {
  try {
    $root = Invoke-WebRequest -Uri $FrontendUrl -UseBasicParsing -TimeoutSec 5
  } catch {
    return [pscustomobject]@{
      Ok = $false
      Detail = "root request failed: $($_.Exception.Message)"
    }
  }

  $cssMatch = [regex]::Match($root.Content, '<link[^>]+rel=["'']stylesheet["''][^>]+href=["'']([^"'']+\.css[^"'']*)["'']')
  if (-not $cssMatch.Success) {
    return [pscustomobject]@{
      Ok = $false
      Detail = "stylesheet link not found"
    }
  }

  $cssUrl = Resolve-FrontendUrl -Path $cssMatch.Groups[1].Value
  try {
    $css = Invoke-WebRequest -Uri $cssUrl -UseBasicParsing -TimeoutSec 5
  } catch {
    return [pscustomobject]@{
      Ok = $false
      Detail = "css request failed: $cssUrl $($_.Exception.Message)"
    }
  }

  if ([int]$root.StatusCode -eq 200 -and [int]$css.StatusCode -eq 200) {
    return [pscustomobject]@{
      Ok = $true
      Detail = "root=200 css=200 $cssUrl"
    }
  }

  return [pscustomobject]@{
    Ok = $false
    Detail = "root=$($root.StatusCode) css=$($css.StatusCode) $cssUrl"
  }
}

function Start-FrontendDev {
  $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
  if (-not $npm) {
    $npm = "npm"
  }

  "" | Set-Content -Path $FrontendOutLog -Encoding UTF8
  "" | Set-Content -Path $FrontendErrLog -Encoding UTF8

  $command = "cd /d `"$FrontendDir`" && set NEXT_PUBLIC_BACKEND_URL=$BackendUrl && `"$npm`" run dev -- --hostname $FrontendHost --port $FrontendPort 1> `"$FrontendOutLog`" 2> `"$FrontendErrLog`""

  $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $processInfo.FileName = "cmd.exe"
  $processInfo.Arguments = "/d /s /c `"$command`""
  $processInfo.WorkingDirectory = $FrontendDir
  $processInfo.UseShellExecute = $false
  $processInfo.CreateNoWindow = $true
  $processInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden

  $process = [System.Diagnostics.Process]::Start($processInfo)
  Write-Host "프론트 dev server launcher PID=$($process.Id)"
  Write-Host "프론트 stdout: $FrontendOutLog"
  Write-Host "프론트 stderr: $FrontendErrLog"
}

$backendReady = Test-BackendReady
if ($backendReady) {
  Write-Host "백엔드 ready: http://127.0.0.1:8000/api/v1/health/live"
} else {
  $backendListening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
  if (-not $backendListening) {
    Start-BackendDev
  } else {
    Write-Host "백엔드 8000 포트는 열려 있지만 아직 ready 상태가 아닙니다. 기존 프로세스가 기동 중인지 확인합니다."
  }

  $backendDeadline = (Get-Date).AddSeconds(45)
  do {
    if (Test-BackendReady) {
      $backendReady = $true
      Write-Host "백엔드 ready: http://127.0.0.1:8000/api/v1/health/live"
      break
    }
    Start-Sleep -Seconds 2
  } while ((Get-Date) -lt $backendDeadline)
}

if (-not $backendReady) {
  Write-Host "백엔드가 45초 안에 ready 상태가 되지 않았습니다."
  Write-Host "백엔드 stdout 마지막 로그:"
  Get-Content -Path $BackendOutLog -Tail 40 -ErrorAction SilentlyContinue
  Write-Host "백엔드 stderr 마지막 로그:"
  Get-Content -Path $BackendErrLog -Tail 80 -ErrorAction SilentlyContinue
  exit 1
}

if (Test-FrontendReady | Select-Object -ExpandProperty Ok) {
  $frontendCheck = Test-FrontendReady
  Write-Host "프론트 ready: $($frontendCheck.Detail)"
} else {
  Stop-FrontendPortOwner
  Start-Sleep -Seconds 2
  Start-FrontendDev

  $frontendDeadline = (Get-Date).AddSeconds($FrontendTimeoutSeconds)
  do {
    Start-Sleep -Seconds 2
    $frontendCheck = Test-FrontendReady
    if ($frontendCheck.Ok) {
      Write-Host "프론트 ready: $($frontendCheck.Detail)"
      break
    }
    Write-Host "프론트 대기 중: $($frontendCheck.Detail)"
  } while ((Get-Date) -lt $frontendDeadline)
}

if (-not $frontendCheck.Ok) {
  Write-Host "프론트가 $FrontendTimeoutSeconds초 안에 ready 상태가 되지 않았습니다."
  Write-Host "프론트 stdout 마지막 로그:"
  Get-Content -Path $FrontendOutLog -Tail 40 -ErrorAction SilentlyContinue
  Write-Host "프론트 stderr 마지막 로그:"
  Get-Content -Path $FrontendErrLog -Tail 40 -ErrorAction SilentlyContinue
  exit 1
}

Write-Host "Lens 데모 서버 실행 요청이 완료되었습니다."
Write-Host "백엔드:  http://127.0.0.1:8000"
Write-Host "프론트:  http://127.0.0.1:3000"
Write-Host "백엔드 로그: logs/backend_dev.out.log, logs/backend_dev.err.log"
Write-Host "프론트 로그: logs/frontend_dev.out.log, logs/frontend_dev.err.log"
Write-Host "상태 확인: powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1"
Write-Host "주의: npm run build 후에는 프론트 dev 서버를 재시작해야 static asset 404를 피할 수 있습니다."
exit 0
