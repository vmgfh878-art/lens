param(
    [string]$TaskName = "Lens yfinance 500 daily append",
    [string]$TaskDescription = "Lens local yfinance 500 parquet daily append and 1D/1W indicator refresh",
    [string]$At = "08:20",
    [string[]]$DaysOfWeek = @("Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
    [bool]$RunAtLogOn = $false,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ScriptPath = Join-Path $Root "scripts\run_yfinance_500_daily_append_automation.ps1"
if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "자동화 스크립트를 찾을 수 없습니다: $ScriptPath"
}

$PowerShell = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -Apply"

Write-Host "task_name=$TaskName"
Write-Host "script=$ScriptPath"
Write-Host "time=$At"
Write-Host "days=$($DaysOfWeek -join ',')"
Write-Host "run_at_logon=$RunAtLogOn"
Write-Host "argument=$Argument"

if ($DryRun) {
    Write-Host "dry_run=true"
    exit 0
}

# 현재 사용자 계정으로 등록한다. 컴퓨터가 켜져 있고 작업 스케줄러가 실행 가능한 상태여야 한다.
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Argument -WorkingDirectory $Root
$Triggers = @(
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DaysOfWeek -At $At
)
if ($RunAtLogOn) {
    $Triggers += New-ScheduledTaskTrigger -AtLogOn
}
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Triggers `
    -Settings $Settings `
    -Description $TaskDescription `
    -Force | Out-Null

$Registered = Get-ScheduledTask -TaskName $TaskName
$Info = Get-ScheduledTaskInfo -TaskName $TaskName

[pscustomobject]@{
    task_name = $Registered.TaskName
    state = $Registered.State
    next_run_time = $Info.NextRunTime
    last_run_time = $Info.LastRunTime
    last_task_result = $Info.LastTaskResult
} | Format-List
