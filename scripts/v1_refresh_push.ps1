# CP255 — 일일 refresh 의 production push 를 "현재 체크아웃 브랜치와 무관하게" 항상 origin/main 으로.
#
# run_v1_unified_refresh_local.ps1 이 이 파일을 dot-source 해서 Invoke-V1ProductionPush 를 호출한다.
# 함수는 self-contained(스크립트 스코프 변수에 의존하지 않음) 라서, 샌드박스 repo 에서 dot-source 후
# 단독으로 분기 로직을 검증할 수 있다(CP255 §4 모의/정상/dry-run).
#
# 설계 배경(CP255 §0): push 블록이 `git commit` 을 "현재 체크아웃 브랜치"에 붙이고 `git push origin main`
# 하던 탓에, 병렬 세션이 main 체크아웃을 feature 로 바꿔둔 동안 6/11~6/17 일일 refresh 8개가 전부
# feature 로 커밋되고 push origin main 은 무동작 → 프로덕션이 8일 정체했다. 운영 서빙 데이터는
# 현재 작업 브랜치와 무관하게 항상 main 으로 가야 한다.

function Invoke-V1ProductionPush {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RunDate,
        [bool]$Apply = $false,
        [bool]$AnyStepFailed = $false,
        [string]$FailedNames = "",
        [string]$LogPath = $null
    )

    function _log {
        param([string]$Message)
        $line = "[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $Message
        Write-Host $line
        if ($LogPath) { Add-Content -Path $LogPath -Value $line -Encoding UTF8 }
    }

    # CP238 — 단계 실패 시 production push 차단 (두 경로 공통 앞단 가드).
    # 이전엔 band/line refresh 가 죽어도 market/product 만 갱신된 채 push 가 진행돼,
    # 예측이 옛 날짜에 정체된 데이터가 운영에 배포되는 사고(2026-06-05). 핵심 단계가
    # 하나라도 FAIL 이면 push 를 막는다.
    if ($Apply -and $AnyStepFailed) {
        _log "CRITICAL: 단계 실패로 production push 차단 (failed: $FailedNames). 운영 데이터 불일치(예측 정체) 방지를 위해 commit/push 를 건너뛴다. 각 단계 stderr 확인 후 재실행하라. working tree 변경 폐기는 git checkout -- backend/data/v1 ."
        return "BLOCKED_STEP_FAILURE"
    }
    if (-not $Apply) {
        return "SKIPPED_DRY_RUN"
    }

    # PowerShell 5.1 quirk 회피: native git 의 stderr(pre-commit 훅/진행 출력 등)를 2>&1 로 머지하면
    # NativeCommandError 로 감싸지고 ErrorActionPreference=Stop 이 종료 오류로 승격시켜, commit 은
    # 됐는데 push 가 건너뛰어지는 사고(2026-06-10, push=WARN_PUSH_ERROR 인데 데이터는 로컬 커밋만 됨).
    # 그래서 (1) 이 블록만 EAP=Continue 로 낮춰 native stderr 가 throw 하지 않게 하고,
    # (2) 2>&1 머지를 제거해 git stderr 는 그대로 흘리며,
    # (3) 성공/실패는 PowerShell 예외가 아니라 git 의 $LASTEXITCODE 로만 판정한다.
    $PushStatus = "WARN_PUSH_ERROR"
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $CurrentBranch = "$(git -C $Root rev-parse --abbrev-ref HEAD)".Trim()
        if ($CurrentBranch -eq "main") {
            # ── 정상 경로: 현재 체크아웃이 main → 지금처럼 직접 add/commit/push (회귀 0) ──
            _log "production git push 시작 (현재 브랜치 main, 직접 경로)"
            git -C $Root add backend/data/v1/*.parquet backend/data/v1/*.json | Out-Null
            git -C $Root diff --cached --quiet
            if ($LASTEXITCODE -ne 0) {
                git -C $Root commit -m "data: daily refresh $RunDate" | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    $PushStatus = "WARN_COMMIT_FAILED"
                    _log "production git commit 실패 (pre-commit 훅/충돌 확인)"
                } else {
                    git -C $Root push origin main | Out-Null
                    if ($LASTEXITCODE -eq 0) {
                        $PushStatus = "PASS"
                        _log "production git push 완료"
                    } else {
                        $PushStatus = "WARN_PUSH_FAILED"
                        _log "production git push 실패 (네트워크/인증 확인)"
                    }
                }
            } else {
                $PushStatus = "NO_CHANGES"
                _log "serving parquet 변경 없음, push 생략"
            }
        } else {
            # ── 견고화 경로: 현재 체크아웃이 non-main → 전용 worktree 로 main 에 직접 올린다 ──
            # 서빙 데이터는 현재 작업 브랜치와 무관하게 항상 origin/main 으로 가야 한다(CP255).
            $wt = Join-Path $Root ".git-refresh-main"
            _log "production git push 시작 (현재 브랜치 '$CurrentBranch' != main → 전용 worktree 로 main 에 직접 커밋: $wt)"
            try {
                # 매 회 add/remove 가 단순·안전. (영속 worktree 는 main 을 거기 묶어 메인 체크아웃을
                # main 으로 되돌릴 때 충돌나므로 쓰지 않는다.) prune 으로 직전 크래시 잔여 등록도 정리.
                git -C $Root worktree prune | Out-Null
                if ((Test-Path -LiteralPath $wt) -and -not (Test-Path -LiteralPath (Join-Path $wt ".git"))) {
                    # 디렉토리는 남았는데 worktree 등록이 아닌 잔여물 → 비우고 새로 만든다.
                    Remove-Item -LiteralPath $wt -Recurse -Force
                }
                if (-not (Test-Path -LiteralPath (Join-Path $wt ".git"))) {
                    git -C $Root worktree add $wt main | Out-Null
                }
                if (-not (Test-Path -LiteralPath (Join-Path $wt ".git"))) {
                    $PushStatus = "WARN_WORKTREE_ADD_FAILED"
                    _log "main worktree 생성 실패 (main 이 다른 곳에 체크아웃됐는지 확인): $wt"
                } else {
                    # 전용 worktree 만 origin/main 최신으로 맞춘다. (reset --hard 는 사용자 작업트리엔
                    # 절대 적용하지 않는다 — 이 worktree 에는 refresh 데이터 외 추적물이 없다.)
                    git -C $wt fetch origin main | Out-Null
                    git -C $wt checkout main | Out-Null
                    git -C $wt reset --hard origin/main | Out-Null

                    # 복사 대상 한정: main 이 tracked 하는 backend/data/v1 산출 파일만 현재 작업트리에서
                    # worktree 로 복사한다. 신규/타브랜치 전용 데이터 유입 차단.
                    $tracked = @(git -C $wt ls-files backend/data/v1/)
                    if ($tracked.Count -eq 0) {
                        $PushStatus = "WARN_NO_TRACKED_FILES"
                        _log "main 의 backend/data/v1 tracked 파일이 없음 — push 중단"
                    } else {
                        $copied = 0
                        $missing = 0
                        foreach ($rel in $tracked) {
                            if (-not $rel) { continue }
                            $src = Join-Path $Root $rel
                            $dst = Join-Path $wt $rel
                            if (Test-Path -LiteralPath $src) {
                                Copy-Item -LiteralPath $src -Destination $dst -Force
                                $copied++
                            } else {
                                $missing++
                                _log "tracked 파일이 현재 작업트리에 없어 복사 건너뜀: $rel"
                            }
                        }
                        _log "worktree 로 복사: $copied 파일 (누락 $missing)"
                        git -C $wt add -- @tracked | Out-Null
                        git -C $wt diff --cached --quiet
                        if ($LASTEXITCODE -ne 0) {
                            git -C $wt commit -m "data: daily refresh $RunDate" | Out-Null
                            if ($LASTEXITCODE -ne 0) {
                                $PushStatus = "WARN_COMMIT_FAILED"
                                _log "main worktree commit 실패 (pre-commit 훅/충돌 확인)"
                            } else {
                                git -C $wt push origin main | Out-Null
                                if ($LASTEXITCODE -eq 0) {
                                    $PushStatus = "PASS"
                                    _log "production git push 완료 (worktree → origin/main)"
                                } else {
                                    $PushStatus = "WARN_PUSH_FAILED"
                                    _log "production git push 실패 (네트워크/인증/비-fast-forward 확인)"
                                }
                            }
                        } else {
                            $PushStatus = "NO_CHANGES"
                            _log "serving 데이터가 origin/main 과 동일, push 생략"
                        }
                    }
                }
            } finally {
                # per-run 정리: worktree 제거(권장). 제거 실패해도 다음 회차 prune/재생성이 흡수.
                if (Test-Path -LiteralPath (Join-Path $wt ".git")) {
                    git -C $Root worktree remove --force $wt | Out-Null
                }
                git -C $Root worktree prune | Out-Null
            }
        }
    } catch {
        $PushStatus = "WARN_PUSH_ERROR"
        _log "git push 오류: $($_.Exception.Message)"
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    return $PushStatus
}
