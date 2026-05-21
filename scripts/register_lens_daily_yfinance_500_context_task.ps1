param(
    [string]$TaskName = "LensDailyYFinance500LocalUpdate",
    [ValidateSet("Logon", "Daily10", "Both")]
    [string]$TriggerMode = "Logon",
    [string]$DailyAt = "10:00",
    [switch]$Register
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$PowerShell = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
$RunnerScript = Join-Path $Root "scripts\run_lens_daily_yfinance_500_context_task.ps1"

if (-not (Test-Path -LiteralPath $PowerShell)) {
    throw "PowerShell 실행 파일을 찾을 수 없습니다: $PowerShell"
}
if (-not (Test-Path -LiteralPath $RunnerScript)) {
    throw "작업 스케줄러 실행 래퍼를 찾을 수 없습니다: $RunnerScript"
}

$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$RunnerScript`""

$Summary = [pscustomobject]@{
    task_name = $TaskName
    trigger_mode = $TriggerMode
    daily_at = $DailyAt
    working_directory = $Root
    execute = $PowerShell
    argument = $Argument
    register = [bool]$Register
}

if (-not $Register) {
    $Summary | Format-List
    Write-Host "dry_run=true"
    Write-Host "실제 등록하려면 -Register를 명시하세요."
    exit 0
}

$Triggers = @()
if ($TriggerMode -eq "Logon" -or $TriggerMode -eq "Both") {
    $Triggers += New-ScheduledTaskTrigger -AtLogOn
}
if ($TriggerMode -eq "Daily10" -or $TriggerMode -eq "Both") {
    $Triggers += New-ScheduledTaskTrigger -Daily -At $DailyAt
}

$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Argument -WorkingDirectory $Root
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Triggers `
    -Settings $Settings `
    -Description "Lens yfinance 500 local price, indicator, and context daily update" `
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
