param(
  [string]$FrontendUrl = "http://127.0.0.1:3000",
  [string]$BackendUrl = "http://127.0.0.1:8000",
  [string]$HostName = "127.0.0.1",
  [int]$Port = 3000,
  [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$FrontendDir = Join-Path $RootDir "frontend"
$LogDir = Join-Path $RootDir "logs"
$OutLog = Join-Path $LogDir "frontend_dev.out.log"
$ErrLog = Join-Path $LogDir "frontend_dev.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Stop-FrontendPortOwner {
  $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
  foreach ($listener in $listeners) {
    $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    if (-not $process) {
      continue
    }

    if ($process.ProcessName -in @("node", "cmd", "npm")) {
      Write-Host "Stopping frontend dev process PID=$($process.Id) name=$($process.ProcessName)"
      Stop-Process -Id $process.Id -Force
    } else {
      Write-Host "Port $Port is owned by $($process.ProcessName) PID=$($process.Id). Not stopping it automatically."
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

function Test-FrontendStatic {
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

  "" | Set-Content -Path $OutLog -Encoding UTF8
  "" | Set-Content -Path $ErrLog -Encoding UTF8

  $command = "cd /d `"$FrontendDir`" && set NEXT_PUBLIC_BACKEND_URL=$BackendUrl && `"$npm`" run dev -- --hostname $HostName --port $Port 1> `"$OutLog`" 2> `"$ErrLog`""

  $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $processInfo.FileName = "cmd.exe"
  $processInfo.Arguments = "/d /s /c `"$command`""
  $processInfo.WorkingDirectory = $FrontendDir
  $processInfo.UseShellExecute = $false
  $processInfo.CreateNoWindow = $true
  $processInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden

  $process = [System.Diagnostics.Process]::Start($processInfo)
  Write-Host "Started frontend dev server launcher PID=$($process.Id)"
  Write-Host "stdout: $OutLog"
  Write-Host "stderr: $ErrLog"
}

Stop-FrontendPortOwner
Start-Sleep -Seconds 2
Start-FrontendDev

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
  Start-Sleep -Seconds 2
  $check = Test-FrontendStatic
  if ($check.Ok) {
    Write-Host "Frontend ready: $($check.Detail)"
    exit 0
  }
  Write-Host "Waiting frontend: $($check.Detail)"
} while ((Get-Date) -lt $deadline)

Write-Host "Frontend dev server did not become ready within $TimeoutSeconds seconds."
Write-Host "Last stdout:"
Get-Content -Path $OutLog -Tail 40 -ErrorAction SilentlyContinue
Write-Host "Last stderr:"
Get-Content -Path $ErrLog -Tail 40 -ErrorAction SilentlyContinue
exit 1
