# CP238 지시서 — Dependency Vulnerability Audit (OWASP A06)

> 작성: 2026-06-06. 트랙: Lens 보안 트랙 (CP238~242). 1순위 CP. ADR-0029 동반 작성.
> 사용자 환경: Windows PowerShell. 한국어 보고서.

---

## 0. 한 줄 목표

`backend/requirements.txt` + `frontend/package.json` 의 모든 의존성에서 알려진 CVE를 자동 검출하고 high/critical은 fix. CI에 게이트 박아 재발 차단.

---

## 1. 진단

| 항목 | 현재 |
|---|---|
| backend CVE 검사 | 한 적 없음 |
| frontend CVE 검사 | 한 적 없음 |
| `.github/workflows/ci.yml` audit job | 없음 (CP237 ci.yml에는 ruff/mypy/pytest/snapshot/tsc/vitest/playwright만) |
| GitHub Dependabot alerts | 미확인 (활성화 필요) |
| 보안 도구 설치 여부 | 0 (pip-audit/safety/bandit/gitleaks 모두 없음 확인됨) |

**OWASP**: A06 Vulnerable & Outdated Components.

---

## 2. 변경 내용

- `backend/requirements-dev.txt` 에 `pip-audit` 추가
- 1회 전체 audit 실행 → high/critical CVE 발견 시 패키지 버전 bump
- `npm audit` 실행 + 자동 fix
- `.github/workflows/ci.yml` 에 audit job 추가 (backend + frontend)
- GitHub Settings → Security → Dependabot alerts on
- ADR-0029 작성

---

## 3. Step 분할

| Step | 내용 | 위험 | 시간 | 자동/수동 |
|---|---|---|---|---|
| 1 | `pip-audit` 의존성 추가 (`backend/requirements-dev.txt`) | 매우낮음 | 5분 | 자동 |
| 2 | `pip-audit` 1회 실행 → `docs/cp238_pip_audit_report.json` 저장 | 매우낮음 (read-only) | 5분 | 자동 |
| 3 | high/critical CVE 발견 시 버전 bump → 재실행 → 0 confirm | 중간 (회귀 가능) | 30분~1h | 반자동 (사용자 확인) |
| 4 | `npm audit --audit-level=moderate --json` 실행 → `docs/cp238_npm_audit_report.json` | 매우낮음 | 5분 | 자동 |
| 5 | `npm audit fix` 자동 가능한 것 처리 → 재확인 | 낮음 | 15분 | 반자동 |
| 6 | 자동 fix 안 되는 high/critical은 수동 (npm `overrides` 또는 직접 bump) | 중간 | 30분 | 수동 |
| 7 | `.github/workflows/ci.yml` audit job 추가 (backend + frontend 둘 다) | 낮음 | 15분 | 자동 |
| 8 | GitHub Settings → Security → Dependabot alerts on (UI) | 매우낮음 | 5분 | **수동** (사용자가 GitHub UI에서) |
| 9 | `docs/cp238_dependency_audit_report.md` 보고서 + ADR-0029 작성 | 매우낮음 | 30분 | 자동 |

---

## 4. 각 Step 정확한 명령 / 코드

### Step 1 — pip-audit 의존성 추가

`backend/requirements-dev.txt` 끝에 추가:

```
# CP238 — 의존성 CVE audit
pip-audit==2.7.3
```

설치:
```powershell
cd C:\Users\user\lens\backend
pip install pip-audit==2.7.3
```

### Step 2 — pip-audit 1회 실행

```powershell
cd C:\Users\user\lens
pip-audit -r backend/requirements.txt `
  --format json `
  --output docs/cp238_pip_audit_report.json `
  --strict
```

**주의**:
- `torch` 가 `requirements.txt` 에 명시 안 됨 (사용자 GPU sm_120 cu128 nightly 별도 설치) → audit 대상에서 자동 제외 ✅
- `--strict` 는 vulnerable 1개라도 있으면 exit code 1 → CI 게이트에 그대로 활용

요약 형식 추가 출력:
```powershell
pip-audit -r backend/requirements.txt --format columns
```

### Step 3 — high/critical fix

`cp238_pip_audit_report.json` 의 `dependencies[].vulns[]` 에서 `fix_versions` 확인.

예시 (가능 시나리오):
```
fastapi==0.111.0  → CVE-2024-xxxxx (high) → fix: 0.111.2
```

수정:
```powershell
# backend/requirements.txt 의 fastapi==0.111.0 → fastapi==0.111.2
# 그리고:
pip install -r backend/requirements.txt --upgrade
pip-audit -r backend/requirements.txt --strict  # 0 confirm
```

**회귀 검증 필수**:
```powershell
pytest backend/tests -q
pytest backend/tests -k "snapshot or characterization or cp223" -q
```

→ 통과해야 commit. 깨지면 bump 되돌리고 다른 버전 시도.

### Step 4 — npm audit

```powershell
cd C:\Users\user\lens\frontend
npm audit --audit-level=moderate --json > ../docs/cp238_npm_audit_report.json
npm audit --audit-level=moderate   # 사람용 요약
```

### Step 5 — npm audit fix

```powershell
cd C:\Users\user\lens\frontend
npm audit fix
npm audit --audit-level=moderate   # 재확인
```

회귀:
```powershell
npx tsc --noEmit
npm run test:unit
npx playwright install --with-deps chromium   # 첫 회만
npm run test:e2e
```

### Step 6 — 수동 fix (자동 fix 실패 시)

`package.json` 에 `overrides` 추가 예:
```json
{
  "overrides": {
    "취약-패키지명": "^안전버전"
  }
}
```

또는 직접 의존성 bump:
```powershell
npm install lightweight-charts@latest   # 예시
```

각 bump 후 `npm audit` + 회귀 검증.

### Step 7 — CI workflow에 audit job 추가

`.github/workflows/ci.yml` 의 `backend` job 끝, `frontend` job 끝에 각각 step 추가:

**backend job 끝에 (`Pytest (CP223 characterization snapshot)` step 다음)**:
```yaml
      - name: pip-audit (CP238 — fail on high/critical)
        run: |
          pip install pip-audit==2.7.3
          pip-audit -r backend/requirements.txt --strict
```

**frontend job 끝에 (`Playwright tests` step 다음)**:
```yaml
      - name: npm audit (CP238 — fail on high/critical)
        run: npm audit --audit-level=high
```

`--audit-level=high` 로 한 이유: moderate 까지 게이트 걸면 false positive 많음. high 부터 실패 처리.

### Step 8 — Dependabot alerts (UI)

사용자가 직접:
1. GitHub.com → `vmgfh878-art/lens` repo
2. Settings → Code security and analysis
3. **Dependabot alerts**: Enable
4. **Dependabot security updates**: Enable (자동 PR)
5. **Dependabot version updates**: Optional (이건 PR 노이즈 많음, 보류 권장)

**자동 X**. 사용자가 GitHub UI에서 직접.

### Step 9 — 보고서 + ADR

`docs/cp238_dependency_audit_report.md` 양식:

```markdown
# CP238 Dependency Audit 보고서

## 결과 요약
- backend pip-audit: 발견 N건 → fix M건 → 잔여 K건 (justification 첨부)
- frontend npm audit: 발견 N건 → fix M건 → 잔여 K건
- CI workflow 추가: backend audit + frontend audit
- Dependabot alerts: enabled (사용자 GitHub UI)

## 변경된 패키지
| 패키지 | 이전 | 이후 | CVE | 근거 |
|---|---|---|---|---|
| (예) fastapi | 0.111.0 | 0.111.2 | CVE-XXXX | high, fix available |

## 잔여 CVE 정당화
- (있으면 각 CVE 어떤 사용 패턴이라 영향 없는지)

## 회귀 검증
- backend pytest: PASS (N tests)
- CP223 snapshot: 0 diff
- frontend tsc + vitest + playwright: PASS

## 산출물
- docs/cp238_pip_audit_report.json
- docs/cp238_npm_audit_report.json
- .github/workflows/ci.yml diff
- docs/adr/0029_dependency_audit_policy.md
```

ADR 양식은 §9 참조.

---

## 5. 회귀 안전망

- **CP223 BE snapshot** (snaptol): 패키지 bump 후 운영 모델 3개 추론 출력 + 주요 API 응답이 float tolerance 안에 들어오는지
- **CP230 FE smoke** (Playwright/Vitest): UI 회귀 0
- Step 3, 6 의 bump 후 즉시 위 두 가지 통과 확인 → 통과해야 commit

---

## 6. 성공 기준 (L8 측정 가능)

- `pip-audit -r backend/requirements.txt --strict` exit 0
- `npm audit --audit-level=high` exit 0
- CI workflow audit job 둘 다 GREEN
- Dependabot alerts active
- `docs/cp238_pip_audit_report.json` + `docs/cp238_npm_audit_report.json` 존재
- 모든 회귀 (pytest + snapshot + tsc + vitest + playwright) PASS

---

## 7. 인터페이스 보존 (L7)

- 패키지 이름은 그대로, 버전만 bump → API contract 영향 0
- 운영 모델 3개 추론 결과 변동 0 (snaptol tolerance)
- frontend props / event 시그니처 영향 0

---

## 8. Lens 특화 (L9)

- **torch CPU/GPU 분리 정책 유지** (CP237) — `torch` 는 `requirements.txt` 에 없어 audit 대상 자동 제외
- **wandb, pandas, pyarrow, scikit-learn, statsmodels** 같은 ML/data 라이브러리 CVE 우선 점검 (자주 발견됨)
- **운영 모델 3개 추론 영향 0** 확인: pytest snapshot
- **daily refresh 영향 0** 확인: `scripts/run_v1_unified_refresh_local.ps1` 한 번 dry-run
- **CI는 CPU만** — Step 7 yaml 작성 시 GPU 의존 path 추가 X

---

## 9. ADR-0029 작성 가이드

파일: `docs/adr/0029_dependency_audit_policy.md`

```markdown
# ADR-0029: Dependency Vulnerability Audit Policy

## Status
Accepted (2026-06-06)

## Context
v1 운영 진입 후에도 backend (FastAPI/pandas/pyarrow/...) 및 frontend (Next.js/React/lightweight-charts/...) 의존성에 알려진 CVE 점검을 한 번도 한 적 없음. OWASP Top 10 A06 무방비.

## Decision
1. **도구 선정**: backend = `pip-audit` (PyPA 공식, PyPI advisory DB 직접 조회). frontend = `npm audit` (npm 내장, ecosystem 표준).
   - rejected: `safety` (commercial DB, free tier 제약), `bandit` (소스 분석이라 의존성 CVE 아님 — 별도 트랙).
2. **CI 게이트 임계값**: backend = `--strict` (모든 vuln 차단). frontend = `--audit-level=high` (moderate 는 noise).
3. **예외 처리**: 잔여 CVE 는 ADR 또는 보고서에 사용 패턴 justification + 모니터링 주기 명시.
4. **Dependabot alerts**: enabled. Dependabot security updates auto PR enabled. version updates 보류.
5. **재발 차단**: CI workflow 에 audit job 박아 PR/push 마다 자동 검사.

## Consequences
- 신규 의존성 추가 시 audit 통과 필수 → PR 속도 약간 저하
- 의존성 bump 잦아짐 → 회귀 안전망 (CP223 snapshot + CP230 smoke) 부담 증가, 다만 자동 → 무리 없음
- 잔여 CVE justification 누적 시 별도 관리 필요 → docs/security/cve_exceptions.md (미래)

## References
- OWASP Top 10 A06
- pip-audit https://github.com/pypa/pip-audit
- npm audit docs
```

---

## 10. 자동 실행 적합도

| Step | 자동 | 사람 확인 필요 |
|---|---|---|
| 1 | ✅ | — |
| 2 | ✅ | — |
| 3 | △ | 버전 bump 후 회귀 검증 결과 사용자 확인 |
| 4 | ✅ | — |
| 5 | △ | npm audit fix 가 breaking change 동반 가능 |
| 6 | △ | overrides 정책 결정 |
| 7 | ✅ | — |
| 8 | ❌ | GitHub UI 직접 (Dependabot enable) |
| 9 | ✅ | — |

---

## 11. 종료 후 commit / 보고

### 권장 commit 분할

```
CP238 Step 1: add pip-audit dev dep
CP238 Step 2-3: pip-audit run + N CVE fixes (backend)
CP238 Step 4-5: npm audit + auto-fix (frontend)
CP238 Step 6: manual fix N CVE via overrides (frontend)
CP238 Step 7: CI workflow audit jobs (backend + frontend)
CP238 report + ADR-0029 (dependency audit policy)
```

(Step 8 Dependabot enable 은 GitHub UI 작업이라 별도 commit 없음 — 보고서에 명시.)

### 보고서 위치
`docs/cp238_dependency_audit_report.md`

### ADR 위치
`docs/adr/0029_dependency_audit_policy.md`

---

**진입 조건**: 없음 (독립 CP).
**다음 CP**: CP239 (Secrets git history).
