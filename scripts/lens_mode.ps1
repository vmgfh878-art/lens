# CP255 — Lens 데이터 모드 스위처 + 상태 대시보드.
#
# 사용자 요구: "복구되면 바로 Supabase, 죽으면 윈도우, 자유롭게 왔다갔다." 전환은 두 축:
#   (1) read 소스  = Render 백엔드의 자동 폴백 (data_backend.use_supabase) — 손 안 대도
#                    Supabase 살면 DB, 죽으면 parquet. 이 스크립트로 제어 안 함(자동).
#   (2) 데이터 공급 = 윈도우 작업 스케줄러(parquet+git) ↔ GitHub Actions 크론(DB).
#                    이 스크립트가 (2)를 전환하고 (1)의 현재 상태를 보여준다.
#
# 사용:
#   powershell -File scripts\lens_mode.ps1                 # 상태만
#   powershell -File scripts\lens_mode.ps1 -Mode windows   # 윈도우 공급 ON (Actions OFF)
#   powershell -File scripts\lens_mode.ps1 -Mode supabase  # 윈도우 공급 OFF (Actions ON)
#   ... -WhatIf                                            # 미리보기(변경 0)

param(
    [ValidateSet("status", "windows", "supabase")]
    [string]$Mode = "status",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TaskName = "Lens unified daily refresh"
$Workflow = "daily_market_sync.yml"   # Phase 4b 에서 생성. 없으면 gh 단계는 건너뜀.

function Get-SupabaseHost {
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) { return $null }
    $line = Get-Content $envFile | Select-String '^SUPABASE_URL='
    if (-not $line) { return $null }
    $url = ($line[0].ToString() -split '=', 2)[1].Trim()
    if (-not $url) { return $null }
    try { return ([System.Uri]$url).Host } catch { return $null }
}

function Get-SupabaseKey {
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) { return $null }
    $line = Get-Content $envFile | Select-String '^SUPABASE_KEY='
    if (-not $line) { return $null }
    return ($line[0].ToString() -split '=', 2)[1].Trim()
}

function Test-SupabaseReachable {
    # 2단계 판정 — DNS 만으론 부족하다(2026-06-18 실측: DNS 복구됐는데 컴퓨트는 521 다운).
    #   1) DNS 해석: paused/삭제면 호스트 DNS 가 사라진다.
    #   2) REST 실 ping: /rest/v1/ 에 apikey 로 GET. 401/200/404 = 서버 살아있음,
    #      5xx(521 등)/타임아웃 = 컴퓨트 다운(unpause 워밍업 중일 수 있음).
    # 자유전환 자동폴백(data_backend)은 실제 SELECT ping 이라 이 판정과 같은 의미.
    $h = Get-SupabaseHost
    if (-not $h) { return @{ host = $null; reachable = $false; reason = "SUPABASE_URL 미설정" } }
    try {
        Resolve-DnsName $h -ErrorAction Stop | Out-Null
    } catch {
        return @{ host = $h; reachable = $false; reason = "DNS 없음 (paused/삭제/장애)" }
    }
    $key = Get-SupabaseKey
    try {
        $resp = Invoke-WebRequest -Uri "https://$h/rest/v1/" -Headers @{ apikey = $key } `
            -TimeoutSec 8 -UseBasicParsing
        return @{ host = $h; reachable = $true; reason = "REST $($resp.StatusCode)" }
    } catch {
        $code = $null
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        if ($code -and $code -ge 200 -and $code -lt 500) {
            return @{ host = $h; reachable = $true; reason = "REST $code (서버 정상)" }
        }
        $detail = if ($code) { "REST $code" } else { "연결 실패/타임아웃" }
        return @{ host = $h; reachable = $false; reason = "$detail (컴퓨트 다운/워밍업)" }
    }
}

function Get-TaskState {
    try { return (Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop).State.ToString() }
    catch { return "NOT_REGISTERED" }
}

function Show-Status {
    $task = Get-TaskState
    $sb = Test-SupabaseReachable
    $supplyWin = ($task -eq "Ready")            # 윈도우 공급 중?
    $effectiveRead = if ($sb.reachable) { "Supabase(DB)" } else { "parquet(자동 폴백)" }

    Write-Host ""
    Write-Host "================ Lens 데이터 모드 상태 ================" -ForegroundColor Cyan
    Write-Host ("  윈도우 공급 (작업 스케줄러) : {0}" -f $task) -ForegroundColor $(if ($supplyWin) { "Green" } else { "DarkGray" })
    Write-Host ("  Supabase 도달성             : {0}  [{1}]" -f $(if ($sb.reachable) { "OK" } else { "불가" }), $sb.reason) -ForegroundColor $(if ($sb.reachable) { "Green" } else { "Yellow" })
    Write-Host ("  Render read (자동 폴백 결과): {0}" -f $effectiveRead) -ForegroundColor Cyan
    if ($sb.host) { Write-Host ("  Supabase host               : {0}" -f $sb.host) -ForegroundColor DarkGray }
    Write-Host "------------------------------------------------------"
    if ($sb.reachable) {
        Write-Host "  권장: Supabase 살아있음 → '-Mode supabase' 로 윈도우 공급 끄고 DB 로 전환 가능." -ForegroundColor Green
    } else {
        Write-Host "  권장: Supabase 불가 → 윈도우 공급(parquet+git) 유지. read 는 자동으로 parquet." -ForegroundColor Yellow
    }
    Write-Host "  (Render env 는 손대지 않아도 됨 — read 는 도달성 따라 자동 전환)" -ForegroundColor DarkGray
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Set-WindowsTask([bool]$Enable) {
    $verb = if ($Enable) { "Enable" } else { "Disable" }
    if (Get-TaskState -eq "NOT_REGISTERED") {
        Write-Host "  [skip] 작업 스케줄러 '$TaskName' 미등록 (register_*.ps1 로 먼저 등록)." -ForegroundColor Yellow
        return
    }
    if ($WhatIf) { Write-Host "  [WhatIf] 윈도우 작업 스케줄러 $verb" -ForegroundColor DarkGray; return }
    if ($Enable) { Enable-ScheduledTask -TaskName $TaskName | Out-Null }
    else { Disable-ScheduledTask -TaskName $TaskName | Out-Null }
    Write-Host "  윈도우 작업 스케줄러 → $verb" -ForegroundColor Green
}

function Set-ActionsWorkflow([bool]$Enable) {
    # GitHub Actions 크론(Phase 4b). gh CLI + 워크플로 존재 시에만.
    $gh = (Get-Command gh -ErrorAction SilentlyContinue)
    $wfPath = Join-Path $Root ".github\workflows\$Workflow"
    if (-not $gh -or -not (Test-Path $wfPath)) {
        Write-Host "  [skip] GitHub Actions 전환은 Phase 4b(워크플로+gh) 후 활성." -ForegroundColor DarkGray
        return
    }
    $verb = if ($Enable) { "enable" } else { "disable" }
    if ($WhatIf) { Write-Host "  [WhatIf] gh workflow $verb $Workflow" -ForegroundColor DarkGray; return }
    & gh workflow $verb $Workflow
    Write-Host "  GitHub Actions '$Workflow' → $verb" -ForegroundColor Green
}

switch ($Mode) {
    "status" { Show-Status }
    "windows" {
        Write-Host "`n→ windows 모드: 윈도우 공급 ON, Actions OFF" -ForegroundColor Cyan
        Set-WindowsTask $true
        Set-ActionsWorkflow $false
        Write-Host "  read 는 자동 폴백 — Supabase 죽어있으면 parquet, 살아있으면 DB." -ForegroundColor DarkGray
        Show-Status
    }
    "supabase" {
        Write-Host "`n→ supabase 모드: 윈도우 공급 OFF, Actions ON" -ForegroundColor Cyan
        $sb = Test-SupabaseReachable
        if (-not $sb.reachable) {
            Write-Host "  [경고] Supabase 도달 불가 ($($sb.reason)). 윈도우를 끄면 데이터 공급이 끊긴다." -ForegroundColor Red
            Write-Host "  복구 전이라면 -Mode windows 로 유지하라. 그래도 전환하려면 -WhatIf 없이 재실행." -ForegroundColor Red
            if (-not $WhatIf) { return }
        }
        Set-WindowsTask $false
        Set-ActionsWorkflow $true
        Write-Host "  read 는 자동 폴백 — Render env 손대지 않아도 DB 로 전환됨." -ForegroundColor DarkGray
        Show-Status
    }
}
