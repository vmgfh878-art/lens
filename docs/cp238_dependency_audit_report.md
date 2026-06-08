# CP238 — Dependency Vulnerability Audit 보고서

작성: 2026-06-08. OWASP Top 10 A06 대응. ADR-0029 동반.

## 0. 한 줄 요약

backend pip-audit + frontend audit-ci 1회씩 실행 → 발견 27 CVE (backend 6 +
frontend 24) → bump fix (axios 21 + pyarrow 1 + python-dotenv 1 + starlette 1) +
allowlist 19 (backend 4 justified + frontend 8 justified + 7 acknowledged) →
CI 게이트 (`pip-audit --strict` + `audit-ci`) 박음. 정직성 lock = 사용자
acknowledgement 서명. 회귀 안전망 모두 GREEN. CP238 closure.

## 1. 핵심 컴포넌트 존재 체크리스트

CP238 은 의존성 audit + 정책 박음, ML 컴포넌트 변경 0 (RevIN/CI aggregate/ticker
emb 등 체크 N/A). 대신 본 CP 핵심 컴포넌트 체크:

- [x] `backend/requirements.txt` 가 bump 반영 (fastapi 0.115.6 / pyarrow 17.0.0 / python-dotenv 1.2.2)
- [x] `requirements-dev.txt` 에 pip-audit==2.7.3 dev dep
- [x] `backend/tests/_snapshot_normalize.py` (CP237.5) 보존 — 회귀 안전망 정상
- [x] `frontend/package.json` next 14.2.35 + audit-ci devDep
- [x] `frontend/audit-ci.json` 신규, 15 GHSA allowlist (8 justified + 7 acknowledged)
- [x] `docs/security/cve_exceptions.md` backend 4 + frontend 15 박제 + 사용자 acknowledgement 서명 박힘 (김지형, 2026-06-08)
- [x] `docs/cp238_grep_evidence.md` backend 4 + frontend 11 grep 결과 박제 (재실행 가능)
- [x] `.github/workflows/ci.yml` backend job 의 `pip-audit` step + frontend job 의 `audit-ci` step 추가
- [x] `docs/adr/0029_dependency_audit_policy.md` 정책 명문화
- [x] backend pip-audit 로컬 `--strict --ignore-vuln × 4`: exit 0
- [x] frontend audit-ci 로컬: exit 0 ("Passed npm security audit")

## 2. 새 테스트 결과

CP238 자체는 새 unit test 추가 없음 (의존성 audit 라 운영 코드 무변경). 안전망
재실행 결과:

```
pytest backend/tests -k "snapshot or characterization or cp223" -q
21 passed, 110 deselected
```

- characterization snapshot 9 endpoint ✅
- drift simulation 11 케이스 (CP237.5) ✅
- 기타 cp223 매치 1 ✅

```
npm run test:unit (vitest)
8 Test Files passed | 1 skipped (9)
166 Tests passed | 4 todo (170)
```

## 3. Dry-run / 시뮬레이션 결과

CP238 은 ML 모델 forward 호출 없음. 대신 audit 도구 dry-run:

- `pip-audit -r backend/requirements.txt --strict --ignore-vuln × 4`: exit 0
  ("No known vulnerabilities found, 4 ignored")
- `npx audit-ci --config audit-ci.json`: exit 0 ("Passed npm security audit"),
  high+ 5 advisories 발견되었으나 allowlist 박혀 통과
- `npx tsc --noEmit`: 0 error (fastapi 0.111 → 0.115 / next 14.2.3 → 14.2.35
  bump 후에도 type 회귀 0)

## 4. 기존 회귀 통과 건수

### 4.1 회귀 안전망 (사용자 명시: CP223 + CP230 + drift sim)

| 안전망 | 결과 |
|---|---|
| CP223 characterization snapshot (9 endpoint) | ✅ 9 passed |
| CP237.5 drift simulation (11 케이스) | ✅ 11 passed |
| CP230 frontend smoke (Vitest) | ✅ 8 files / 166 passed |
| CP237 tsc | ✅ 0 error |

### 4.2 backend pytest 전체 (참고)

```
pytest backend/tests --ignore=backend/tests/test_services.py -q
118 passed, 11 failed, 2 skipped
```

11 failed 는 CP237.5 보고서 §4.2 에 명시한 **pre-existing** 동일 (test_api.py
prediction_* 7개 + test_market_data_providers 1개 + test_product_prediction_history
3개). CP238 변경 (의존성 bump + audit) 으로 발생한 새 회귀 0. CP237.5 후 동일
카운트 유지.

## 5. 진행 중 발견된 함정 6개

### F1. 지시서의 `backend/requirements-dev.txt` 경로 오류

실제 파일은 루트 `requirements-dev.txt`. 11ee153 commit (ADR-0014) 도 루트
사용 명시. → 루트 파일에 pip-audit 추가.

### F2. Windows cp949 + utf-8 한국어 주석 충돌

`backend/requirements.txt` 의 한국어 주석 (torch 별도 설치 안내) 이 cp949 locale
에서 pip-audit 의 `pip_requirements_parser` decode fail. → `PYTHONUTF8=1` 강제.
CI yml 에도 명시.

### F3. fastapi 0.115.x 의 starlette pin (`<0.39.0,>=0.37.2`) → starlette 단독 1.0.1 핀 conflict

사용자 옵션 B 의도 (실제 위험만 fix + 회귀 폭 적게) 와 충돌. starlette host
header CVE 만 fix 하려면 fastapi 0.136.3 (latest) 까지 jump 필요. R2 발동 →
사용자 옵션 B1 재결정 (실제 위험 acknowledged + grep justification, 회귀 폭 적게).

### F4. starlette 0.41.3 에서 새 CVE 발견 (GHSA-7f5h-v6xp-fcq8, FileResponse Range DoS)

0.37.2 에서는 안 잡혔던 CVE 가 0.41.3 에서 나타남 (advisory DB 갱신). 추가 grep
(`FileResponse|StaticFiles|StreamingResponse|send_file`) → 0 hit → justify.

### F5. next 14.2.35 patch bump 후에도 14+ advisories 잔여

14.2.x 가 maintenance branch 라 advisory backport 0. fix 권장 `next@16.2.7
breaking change`. 사용자 B 결정 ("14.2.35 = 모든 fix") 의 전제와 다름. R2 발동
→ 사용자 B'-acknowledged (8 justified + 7 acknowledged + 정직성 lock) 결정.

### F6. .gitignore 의 `test_cp*.py` 패턴

CP237.5 에서도 발견. CP238 에선 영향 없음 (test 파일 신규 0). 하지만 향후 CP
별 영구 안전망 test 추가 시 cp prefix 안 쓰는 정책 일관.

## 6. 산출물

### 신규
- `docs/cp238_pip_audit_report.json` (backend audit baseline)
- `docs/cp238_npm_audit_report.json` (frontend audit baseline, axios fix 후 + next 14.2.35 시점)
- `docs/cp238_grep_evidence.md` (backend 4 + frontend 11 grep 명령 + 결과 박제)
- `docs/security/cve_exceptions.md` (backend 4 + frontend 15 CVE 별 justification/acknowledgement + 사용자 서명)
- `docs/adr/0029_dependency_audit_policy.md` (정책 ADR)
- `frontend/audit-ci.json` (allowlist 15 GHSA)
- `docs/cp238_dependency_audit_report.md` (본 보고서)

### 수정
- `requirements-dev.txt` (pip-audit==2.7.3 추가)
- `backend/requirements.txt` (fastapi 0.111.0 → 0.115.6 / pyarrow 16.1.0 → 17.0.0 / python-dotenv 1.0.1 → 1.2.2)
- `frontend/package.json` (next 14.2.3 → 14.2.35, audit-ci devDep)
- `frontend/package-lock.json` (axios 1.7.x → 1.17.0 transitive + 위 bump 반영)
- `.github/workflows/ci.yml` (backend job 에 pip-audit step / frontend job 에 audit-ci step)

운영 코드 (`backend/app/`, `frontend/src/`) / ML 모델 / daily refresh cron / API
contract / 응답 schema 0 수정.

## 7. 인터페이스 보존

| 항목 | 상태 |
|---|---|
| API contract (응답 schema) | 0 수정 (CP237.5 drift-resilient snapshot 9/9 PASS 로 검증) |
| 모델 forward 인터페이스 | 0 수정 |
| dataloader / calendar features | 0 수정 |
| daily refresh cron (`run_v1_unified_refresh_local.ps1`) | 0 수정 |
| 운영 모델 3 개 (CP210 / CP153 / CP178) 응답 정확성 | 유지 |
| frontend props / event 시그니처 | 0 수정 (tsc 0 error / vitest 166 passed) |
| CI workflow의 기존 `--ignore=backend/tests/test_services.py` 패턴 | 보존 |

## 8. commit 이력 (10 commit)

```
4bee195 CP238 Step 1: add pip-audit dev dep
1f30632 CP238 Step 2: pip-audit run → 6 CVE found in 3 packages
4c5c75d CP238 Step 3: bump fastapi/python-dotenv/pyarrow + ignore starlette CVE (4 acknowledged)
e169ac9 CP238 Step 4-5: npm audit + auto-fix (frontend)
d8f5cd4 CP238 Step 6a: frontend CVE grep evidence (Lens usage patterns)
b868b5e CP238 Step 6b: cve_exceptions.md frontend 섹션 (8 justified + 7 acknowledged)
424fba2 CP238 Step 6c: next 14.2.35 + audit-ci allowlist + CI workflow gate
62e0630 CP238 Step 6d: ADR-0029 (dependency audit policy, backend + frontend)
ccc82c2 CP238 Step 7: acknowledgement 서명 (김지형, 2026-06-08) + B3-7 보강
<본 commit> CP238 Step 9: report + ADR-0029 closure 보강 (Step 7/8 결과 반영)
```

본 commit 이 11번째 + closure.

## 9. Step 진행 정리

| Step | 결과 | 비고 |
|---|---|---|
| 1 pip-audit dev dep | ✅ 4bee195 | requirements-dev.txt 루트 (지시서 오류 정정) |
| 2 pip-audit run | ✅ 1f30632 | 6 CVE / 3 packages (PYTHONUTF8=1 필요) |
| 3 backend CVE fix (R3) | ✅ 4c5c75d | 사용자 B 선택. 4 ignore + 3 bump |
| 4 npm audit | ✅ e169ac9 | 3 vulns baseline (axios 21 + next 11+ + postcss 1) |
| 5 npm audit fix auto | ✅ (e169ac9 묶음) | axios 21 → 0 transitive |
| 6 수동 fix (R3) | ✅ d8f5cd4 / b868b5e / 424fba2 / 62e0630 | 사용자 B'-acknowledged. next 14.2.35 + 15 allowlist + ADR |
| 7 acknowledgement 서명 (R3) | ✅ ccc82c2 | 사용자 "김지형 20260608 너 믿고 서명한다" + agent 세 번 검수 후 박음 + B3-7 browser cache 영향 보강 |
| 8 Dependabot UI (R3) | ⚠️ 사용자 응답 `"done (or skip)"` — agent 가 외부 GitHub UI 상태 확인 불가. 사용자 본인이 활성화/보류 결정 박힘 | ADR-0029 §7 정책은 enable 권장 |
| 9 보고서 + ADR closure | ✅ 본 commit | |

## 10. 다음 CP

CP239 (Secrets git history scan, gitleaks). 진입 조건 (CP237.5 완료) 충족.
CP238 의 audit 도구 + 정직성 lock 정책이 CP239 에도 referen​ce 됨.
