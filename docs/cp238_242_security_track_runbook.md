# Lens 보안 트랙 Runbook (CP237.5 + CP238 ~ CP242)

> 작성: 2026-06-06. 단일 진입점. **다음 세션 명령 하나만 전달하면 agent 가 이걸 읽고 CP237.5 → CP238 부터 순차 자동 진행**.
> 2026-06-06 업데이트: CP223 snapshot 8 개가 daily refresh 의 data drift 에 깨져 있음 확인. **CP237.5 (drift-resilient snapshot 재설계) 를 prerequisite 으로 추가**. 그 다음 CP238 진입.

---

## 🚀 사용자가 다음 세션에 던질 명령 (그대로 복붙)

````
Lens 보안 트랙 진행. 작업 디렉터리 C:\Users\user\lens. 메인 runbook: docs\cp238_242_security_track_runbook.md. 진입 순서: CP237.5 → CP238 → CP239 → CP240 → CP241 → CP242. 지시서 6개: docs\cp237_5_snapshot_drift_resilient_instructions.md + docs\cp238_*_instructions.md ~ docs\cp242_*_instructions.md.

다음 규칙을 무조건 지킨다. 어떤 경우에도 우회/생략 X:

[R1] 시작 전 Checkpoint (runbook §5). main branch / clean tree / 도구 가용성 (.\.venv\Scripts\ 직접 호출) / CP223 snapshot 상태 확인. CP223 RED 면 즉시 CP237.5 진입 (이게 prerequisite). FE 테스트는 cd frontend; npm run test:unit. Checkpoint RED 있으면 사용자에게 보고 후 의논.

[R2] 진행 중 선택 애매한 거 / 결론 둘 이상으로 갈리는 거 / 지시서에 명시 안 된 함정 / 운영 영향 가능성 있는 결정 / 의심 신호는 무조건 멈추고 사용자에게 묻는다. "이 정도면 알아서 가도 되겠지" 절대 금지. 자율 판단 금지. 의심되면 멈춤 → 상황 + 옵션 + 권장안 보고 → 응답 대기.

[R3] runbook §3 사람 확인 필요 Step 9 개에서는 무조건 멈춤:
  CP238 Step 8 (Dependabot UI), CP238 Step 3·6 (버전 bump 후 회귀),
  CP239 Step 3a·3b·3c (provider rotation / dashboard env / history rewrite),
  CP240 Step 5 (CSP 조정), CP240 Step 8·9 (외부 검증 사이트),
  CP242 Step 2 (Render dashboard CORS 확인), CP242 Step 5 (rate limit 도입 결정)

[R4] 각 Step 끝마다 commit (지시서 §11 양식). CP 완료 시 ADR + 보고서. 회귀 안전망 (CP237.5 후 CP223 snapshot + CP230 FE smoke) 깨지면 즉시 중단 + revert 제안.

[R5] 환경 = Windows PowerShell 5.1. ?? / ?. / ??= 안 됨. 한 줄 여러 명령은 ; 만. venv tool 은 .\.venv\Scripts\<tool>.exe 직접 호출. pytest 항상 --ignore=backend\tests\test_services.py. pytest env: $env:PYTHONPATH=".;backend"; $env:LENS_FORCE_LOCAL="1".

[R6] 다른 세션에서 사용자가 동시에 작업 중 가능 (예: 최근 "503 해결" / refresh 핫픽스). 시작 전 git fetch 후 last commit 다시 확인. push 전에도 fetch. 충돌 신호 있으면 즉시 멈춤 → 사용자에게 알리고 어떻게 할지 의논.

진행 보고는 runbook §8 양식. 시작 시점 첫 응답은 runbook §11 template.

CP237.5 부터 시작.
````

(필요 시 "CP238부터" 같이 시작 CP 지정 가능. 단 §4 의존성 위반 X.)

---

## 1. 진행 원칙 (Fowler/Beck/Feathers 정합, CP221~237 동일)

1. **CP238 → CP239 → CP240 → CP241 → CP242 순차**. 단 CP238/239 는 독립이라 병렬 가능 (agent 가 한 번에 둘 다 시작해도 무방).
2. 각 CP 는 해당 지시서 (`docs/cp{NN}_*_instructions.md`) 따라 **Step 단위** 진행.
3. **Step 완료마다 commit** (commit message 는 각 지시서 §11 양식).
4. **CP 완료 시 ADR + 보고서** 작성 + commit.
5. **사람 확인 필요 Step (§3) 에서 무조건 멈춤** → 사용자에게 묻고 응답 받고 진행.
6. **회귀 안전망 깨지면 즉시 중단** — 그 Step revert 후 사용자 보고.
7. 리팩토링 트랙 원칙 유지: behavior 먼저 보존 (테스트), structure 그 다음. 추출 순서: 순수함수 → I/O → 상태.

---

## 2. CP 목록 + 지시서 경로 + 진입 조건

| CP | 카테고리 | OWASP | 지시서 | 진입 조건 |
|---|---|---|---|---|
| **CP237.5** | **Snapshot drift-resilient 재설계 (prerequisite)** | — | `docs/cp237_5_snapshot_drift_resilient_instructions.md` | 없음. CP238 진입 전 필수. CP223 RED 라 |
| CP238 | Dependency CVE audit | A06 | `docs/cp238_dependency_audit_instructions.md` | CP237.5 완료 (CP223 GREEN) |
| CP239 | Secrets history (gitleaks) | A02 | `docs/cp239_secrets_history_instructions.md` | CP237.5 완료. **Step 3 rotation 은 사람 수동** |
| CP240 | HTTP 보안 헤더 + CSP | A05 | `docs/cp240_security_headers_instructions.md` | CP237.5 + CP230 FE smoke GREEN |
| CP241 | Input validation (Pydantic) | A03 | `docs/cp241_input_validation_instructions.md` | CP237.5 GREEN |
| CP242 | CORS + rate limit + 트랙 종료 | A04/A05 | `docs/cp242_cors_rate_limit_instructions.md` | CP235 Pydantic Settings 존재 (확인됨) |

---

## 3. 사람 확인 필요 Step (절대 자동 X)

agent 가 아래 Step 에 도달하면 **무조건 멈추고 사용자에게 alarm**:

| CP | Step | 이유 |
|---|---|---|
| CP238 | Step 8 (Dependabot enable) | GitHub UI 직접 클릭 필요 |
| CP238 | Step 3, 6 (버전 bump 후 회귀) | 회귀 결과 사용자 검토 권장 |
| **CP239** | **Step 3a (provider 콘솔 rotation)** | **각 provider 콘솔에서 사용자가 새 key 발급/폐기** |
| **CP239** | **Step 3b (Render/Vercel env 교체)** | **dashboard 직접 입력** |
| **CP239** | **Step 3c (history rewrite force push)** | **돌이킬 수 없는 작업, 결정 필요** |
| CP240 | Step 5 (CSP 조정) | 시각 회귀 가능 — 사용자 검증 |
| CP240 | Step 8, 9 (securityheaders.com / Mozilla Observatory) | 외부 URL 사용자 입력 |
| **CP242** | **Step 2 (Render dashboard CORS env 확인)** | **dashboard 직접 확인** |
| **CP242** | **Step 5 (rate limit 도입 결정)** | **사용자 결정 (도입 / 보류)** |

**Alarm 양식 (agent 가 출력)**:

```
⏸️ CP{NN} Step {N} 사람 확인 필요

Step 내용: <한 줄 요약>
이유: <왜 자동 안 되는지>
필요 액션: <사용자가 할 일 1~3줄>

응답 양식: "{NN} {N} done" (완료) / "{NN} {N} skip" (스킵) / "{NN} {N} <설명>" (정보 전달)

응답까지 대기 중...
```

---

## 4. CP 의존성 그래프

```
CP223 (BE snapshot, 기존) ────┬───────────────┐
                              │               │
CP230 (FE smoke, 기존) ──────┼───┐           │
                              │   │           │
CP235 (Pydantic Settings) ────┼───┼───────────┼───┐
                              │   │           │   │
                              ▼   ▼           ▼   ▼
       CP238 ─── CP239 ─── CP240 ── CP241 ─── CP242 ─── 종료
       (독립)    (독립)    (CP223+230)  (CP223)  (CP235)
```

- **병렬 가능**: CP238 / CP239 (둘 다 독립)
- **순차 강제**: CP240 → CP241 → CP242 (각각 다른 안전망 의존, 또 CP242 가 최종 종합)
- **권장**: 순차 진행 (병렬 시 commit 충돌 위험)

---

## 5. 시작 전 사용자 확인 사항 (Checkpoint)

agent 가 CP238 시작 직전에 다음 5개 확인 후 시작:

```powershell
# Checkpoint 1: main branch + clean working tree
git -C C:\Users\user\lens status --short
# → 빈 출력 또는 untracked 만 (modified 0)

git -C C:\Users\user\lens branch --show-current
# → main

# Checkpoint 2: 마지막 commit 이 CP237 또는 그 이후
git -C C:\Users\user\lens log --oneline -1
# → CP237 또는 data refresh

# Checkpoint 3: CP222 도구 동작
cd C:\Users\user\lens
ruff --version
pytest --version
pre-commit --version

# Checkpoint 4: CP223 snapshot 통과
pytest backend/tests -k "snapshot or characterization or cp223" -q

# Checkpoint 5: CP230 FE 테스트 통과
cd frontend && npm run test:unit
```

위 5개 다 GREEN 이면 시작. 하나라도 깨지면 사용자에게 alarm.

---

## 6. 종료 기준

### CP238~242 모두 GREEN 일 때

- `docs/cp242_security_track_summary.md` 작성 완료
- `docs/adr/0029_*.md` ~ `docs/adr/0033_*.md` 5장 작성 완료
- CI workflow 모든 job GREEN (audit + gitleaks 신규 포함)
- securityheaders.com **A 등급** 확인 (`https://lens-ten-delta.vercel.app` + backend URL)
- pip-audit / npm audit 0 high+
- gitleaks 0 finding
- CP223 snapshot 0 diff
- CP230 smoke PASS

---

## 7. 비상 시 (Rollback)

각 CP 가 **commit 단위 분리** 되어있으니 한 줄 복구 가능:

```powershell
# 특정 Step 만 revert
git revert <commit-hash>

# CP 전체 revert (여러 commit)
git revert <마지막-commit>..<첫번째-commit>

# 가장 안전한 절차:
# 1. 어떤 commit 부터 깨졌는지 확인
git log --oneline -20
# 2. 그 직전 commit 까지 reset
git reset --hard <안전-commit>
git push --force-with-lease origin main   # ← 신중. 본인만 작업하니 OK
```

각 ADR 에 **rollback 절차** 명시 (특히 CP239 rotation 의 backup tag).

---

## 8. 진행 보고 양식

agent 가 사용자에게 보내는 표준 format:

### 8.1 Step 완료 시 (1줄)
```
✅ CP{NN} Step {N}: {한줄 요약}. commit {hash}.
```

예:
```
✅ CP238 Step 2: pip-audit 실행 완료. 발견 3건 (high 1 / moderate 2). commit a1b2c3d.
```

### 8.2 CP 완료 시 (블록)
```
🎉 CP{NN} 완료
- Step 진행: N개 commit
- 회귀: pytest PASS / snapshot 0 diff / smoke PASS
- 산출물: docs/cp{NN}_*_report.md, docs/adr/00{NN}_*.md
- Next: CP{NN+1} 진입
```

### 8.3 사람 확인 필요 시 (블록)
§3 양식.

### 8.4 회귀 깨질 시 (긴급)
```
🔴 CP{NN} Step {N} 회귀 감지 → 중단

깨진 검증:
- <어떤 테스트>: <에러 요약>

조치:
- 해당 Step commit revert 권장
- 사용자 확인 후 진행

응답 양식: "{NN} {N} revert" (revert) / "{NN} {N} keep" (유지) / "{NN} {N} debug" (디버그 도움)
```

### 8.5 트랙 종료 시 (전체)
```
🎉🎉 Lens 보안 트랙 (CP238~242) 완료

| CP | 결과 |
|---|---|
| CP238 | ✅ pip-audit 0, npm audit 0, Dependabot on |
| CP239 | ✅ gitleaks clean, pre-commit + CI hook |
| CP240 | ✅ securityheaders.com A, Playwright PASS |
| CP241 | ✅ TickerStr applied, negative test 6+ |
| CP242 | ✅ CORS regex, rate limit 60/min, 트랙 종합 보고서 |

산출물:
- 5 ADR (0029~0033)
- 5 보고서 (cp238~cp242)
- 1 종합 보고서 (cp242_security_track_summary)
- 신규 파일: <목록>
- 수정 파일: <목록>

CI 게이트: 5 신규 job 추가 (모두 GREEN)

OWASP Top 10 v1 대응 100%. 다음 트랙은 v2 Auth 진입 시.
```

---

## 9. 산출물 위치 정리

### 지시서 (이 트랙)
- `docs/cp238_dependency_audit_instructions.md`
- `docs/cp239_secrets_history_instructions.md`
- `docs/cp240_security_headers_instructions.md`
- `docs/cp241_input_validation_instructions.md`
- `docs/cp242_cors_rate_limit_instructions.md`

### 보고서 (트랙 진행 중 작성)
- `docs/cp238_dependency_audit_report.md`
- `docs/cp239_secrets_history_report.md`
- `docs/cp240_security_headers_report.md`
- `docs/cp241_input_validation_report.md`
- `docs/cp242_cors_rate_limit_report.md`
- `docs/cp242_security_track_summary.md` (종합)

### ADR (트랙 진행 중 작성)
- `docs/adr/0029_dependency_audit_policy.md`
- `docs/adr/0030_secrets_rotation_policy.md`
- `docs/adr/0031_security_headers_csp.md`
- `docs/adr/0032_input_validation_pattern.md`
- `docs/adr/0033_cors_rate_limit_security_track_close.md`

### 신규 코드
- `backend/app/core/security_headers.py` (CP240)
- `backend/app/core/validators.py` (CP241)
- `backend/app/core/rate_limit.py` (CP242, 도입 시)
- `backend/tests/test_cp241_input_validation.py`
- `backend/tests/test_cp242_rate_limit.py` (도입 시)
- `frontend/next.config.js` 또는 `.mjs` 수정 (CP240 headers)
- `docs/security/rotation_runbook.md` (CP239)
- `docs/cp238_pip_audit_report.json` (CP238 산출)
- `docs/cp238_npm_audit_report.json` (CP238 산출)
- `docs/cp239_gitleaks_report.json` (CP239 산출)

### 수정 파일
- `backend/app/main.py` (middleware add)
- `backend/app/config/settings.py` (CorsConfig regex, CP242)
- `backend/requirements.txt` (slowapi, CP242)
- `backend/requirements-dev.txt` (pip-audit, CP238)
- `.github/workflows/ci.yml` (audit + gitleaks job)
- `.pre-commit-config.yaml` (gitleaks hook)
- `.gitignore` (CP238~242 화이트리스트 — 이미 박힘)

---

## 10. 시간 예상

| CP | 자동 | 수동 (사람 확인 포함) | 합계 |
|---|---|---|---|
| CP238 | 1~1.5h | 0.5h (Dependabot UI + bump 검증) | 1.5~2h |
| CP239 | 1h | 0~2h (rotation 발견 시) | 1~3h |
| CP240 | 1.5h | 0.5h (CSP + securityheaders 확인) | 2h |
| CP241 | 2~3h | 0h | 2~3h |
| CP242 | 1~1.5h | 0.5h (Render dashboard + 도입 결정) | 1.5~2h |
| **합계** | **6.5~9h** | **1.5~3h** | **8~12h** |

---

## 11. agent 시작 시 첫 응답 양식 (template)

새 세션 시작 시 agent 가 사용자에게 보낼 첫 응답:

```
Lens 보안 트랙 시작 준비.

읽음:
- docs/cp238_242_security_track_runbook.md
- docs/cp238_*_instructions.md (5개)

Checkpoint 5/5:
✅ main branch / clean tree
✅ 마지막 commit: <hash>
✅ ruff/pytest/pre-commit 설치 확인
✅ CP223 snapshot PASS
✅ CP230 FE 테스트 PASS

진행 계획:
1. CP238 (Dependency audit) — 자동 가능, 1~2h
2. CP239 (Secrets) — Step 3 발견 시 사용자 alarm
3. CP240 (보안 헤더) — Step 8/9 사용자 검증 필요
4. CP241 (Input validation) — 자동
5. CP242 (CORS + rate limit + 종료) — Step 2/5 사용자 결정

지금 CP238 Step 1 시작합니다. 진행 보고는 Step 단위.
```

---

## 12. 한 줄 요약

**다음 세션에 §0 명령 하나만 던지면, agent 가 이 runbook + 5개 지시서를 읽고 CP238~CP242 를 8~12시간 안에 자동 진행. 사람 확인 필요 Step (§3) 에서만 멈춤. 회귀 깨지면 즉시 중단 + 보고.**
