# CP239 지시서 — Secrets Git History 점검 + Rotation 정책 (OWASP A02)

> 작성: 2026-06-06. 트랙: Lens 보안 트랙. ADR-0030 동반.
> 사용자 환경: Windows PowerShell. 한국어 보고서.

---

## 0. 한 줄 목표

git history 전체에 secret (API key / 토큰 / 비밀번호) 박혔는지 자동 검출. 발견 시 즉시 rotation. pre-commit + CI 이중 게이트로 재발 차단.

---

## 1. 진단

| 항목 | 현재 |
|---|---|
| `.gitignore` 에 `.env` | ✅ 박힘 (`.env`, `.env.local`, `*.env`) |
| git history 전수 검사 | 한 적 없음 |
| 발견 시 rotation 절차 | 없음 (runbook 없음) |
| pre-commit secrets hook | 없음 |
| CI secrets job | 없음 |
| 사용 중 secrets | `SUPABASE_KEY` / `SUPABASE_URL` / `FRED_API_KEY` / `FMP_API_KEY` / (필요 시) `EODHD_API_KEY` / `WANDB_API_KEY` |

**OWASP**: A02 Cryptographic Failures (secrets 평문 노출).

---

## 2. 변경 내용

- `gitleaks` 설치 (Windows: scoop 또는 직접 binary)
- 전체 history 1회 스캔 → `docs/cp239_gitleaks_report.json`
- 발견 시: **(a) provider 콘솔에서 key rotate → (b) 코드/env 교체 → (c) history rewrite 신중 결정**
- `pre-commit` 에 gitleaks hook 추가
- `.github/workflows/ci.yml` 에 gitleaks job 추가
- `docs/security/rotation_runbook.md` 작성 (각 secret 어디서 발급/rotate)
- ADR-0030 작성

---

## 3. Step 분할

| Step | 내용 | 위험 | 시간 | 자동/수동 |
|---|---|---|---|---|
| 1 | gitleaks 설치 (Windows) | 매우낮음 | 10분 | 반자동 (scoop install 명령) |
| 2 | `gitleaks detect --source . --report-path docs/cp239_gitleaks_report.json --no-banner` 전체 history 스캔 | 매우낮음 (read-only) | 5분 | 자동 |
| 3 | **발견 0이면**: ADR + 보고서 + Step 5~8 로 점프. **발견 시**: rotation 절차 (3a~3c) | 매우낮음 또는 🔴 큼 | 0 ~ 2h | — |
| 3a | (발견 시) 해당 provider 콘솔에서 secret 폐기 + 새 key 발급 | 🔴 큼 (실 서비스 영향) | 10분/provider | **수동** |
| 3b | (발견 시) 로컬 `.env` + Render dashboard + Vercel dashboard 환경변수 교체 | 중간 | 15분 | **수동** |
| 3c | (발견 시) history rewrite 결정 → `git filter-repo` 또는 BFG. 1인 프로젝트라 가능하지만 GitHub 캐시 + fork 가 secret 보존할 수 있음 → rotation 이 진짜 방어, history rewrite 는 흔적 지우기 | 🔴 큼 (force push) | 1h | **수동** |
| 4 | rotation 완료 후 `gitleaks detect` 재실행 → 0 confirm (현재 working tree 기준) | 매우낮음 | 5분 | 자동 |
| 5 | `pre-commit` 에 gitleaks hook 추가 (`.pre-commit-config.yaml`) | 낮음 | 15분 | 자동 |
| 6 | pre-commit hook 검증: 임시 `test_fake.env` 에 가짜 key 박고 commit 시도 → blocked 확인 → 즉시 폐기 | 낮음 | 10분 | 자동 |
| 7 | `.github/workflows/ci.yml` gitleaks job 추가 (PR + push) | 낮음 | 15분 | 자동 |
| 8 | `docs/security/rotation_runbook.md` 작성 (각 provider 발급/rotate 절차) | 낮음 | 30분 | 자동 |
| 9 | `docs/cp239_secrets_history_report.md` + ADR-0030 | 매우낮음 | 30분 | 자동 |

---

## 4. 각 Step 정확한 명령 / 코드

### Step 1 — gitleaks 설치 (Windows)

옵션 A (scoop):
```powershell
scoop install gitleaks
```

옵션 B (직접 binary):
1. https://github.com/gitleaks/gitleaks/releases/latest 에서 `gitleaks_*_windows_x64.zip` 다운로드
2. 압축 풀어 `gitleaks.exe` 를 PATH 가 잡힌 위치에 (예: `C:\Users\user\bin\`)
3. 새 PowerShell 창에서 `gitleaks version` 확인

옵션 C (Go 설치돼 있으면):
```powershell
go install github.com/gitleaks/gitleaks/v8@latest
```

### Step 2 — 전체 history 스캔

```powershell
cd C:\Users\user\lens
gitleaks detect `
  --source . `
  --report-path docs/cp239_gitleaks_report.json `
  --report-format json `
  --no-banner `
  --verbose
```

추가로 사람용 요약:
```powershell
gitleaks detect --source . --no-banner   # exit code 0 (clean) / 1 (found)
```

`gitleaks` 가 자체 룰셋 (AWS, Stripe, GitHub PAT, Slack, JWT, Supabase 등 다수) 적용. Lens 사용 secret 중 `SUPABASE_KEY` (`eyJ...` JWT 형태) 는 JWT 룰에 잡힘.

### Step 3 — 발견 시 분기

**발견 0**:
```
{"Description": "...", "RuleID": "...", "File": "...", ...}
```
이 라인이 0개 → ADR + 보고서로 점프.

**발견 시 (예시 시나리오)**:
```json
{
  "Description": "Identified a Supabase service key",
  "File": "backend/scripts/test_cp123.py",
  "Commit": "abc123",
  "Author": "...",
  "Date": "2026-04-15"
}
```

→ 즉시 **3a Rotation 먼저** 실행.

### Step 3a — Rotation (provider 콘솔, 수동)

각 provider 별:

| Provider | 콘솔 URL | 절차 |
|---|---|---|
| Supabase | https://supabase.com → Project → Settings → API | "Reset service_role key" → 새 key 복사 |
| FRED | https://fred.stlouisfed.org → My Account → API Keys | 옛 key 삭제 → 새 발급 |
| FMP (Financial Modeling Prep) | https://site.financialmodelingprep.com → Dashboard → API Keys | regenerate |
| EODHD | https://eodhd.com → Dashboard | regenerate |
| Weights & Biases | https://wandb.ai → Settings → API keys | revoke + 새 발급 |

### Step 3b — 환경변수 교체

로컬:
```powershell
# backend/.env 수정
notepad C:\Users\user\lens\backend\.env
```

Render dashboard:
1. https://dashboard.render.com → Services → lens-backend → Environment
2. 해당 env var 교체 → Save → 자동 redeploy

Vercel dashboard:
1. https://vercel.com → lens project → Settings → Environment Variables
2. 해당 var 교체 → Redeploy 트리거

검증:
```powershell
# 로컬 backend 띄워서 새 key 로 동작 확인
cd C:\Users\user\lens\backend
uvicorn app.main:app --reload --port 8000
# 다른 창에서:
curl http://127.0.0.1:8000/api/v1/health/live
curl http://127.0.0.1:8000/api/v1/health/ready
```

### Step 3c — history rewrite (신중 결정)

옵션 A (git filter-repo, 권장):
```powershell
pip install git-filter-repo
cd C:\Users\user\lens
git tag backup-pre-rewrite   # 백업
git filter-repo --invert-paths --path backend/scripts/test_cp123.py   # 예시
```

옵션 B (BFG Repo-Cleaner):
```powershell
# bfg.jar 다운로드 후
java -jar bfg.jar --replace-text passwords.txt   # passwords.txt 에 매칭 패턴
```

Force push (위험):
```powershell
git push --force-with-lease origin main
```

**경고**:
- GitHub 의 cached views, PR diff, raw URL 은 force push 후에도 일정 시간 secret 노출 가능
- 협업자 (1인이라 없지만) 가 fork 했으면 그쪽에 남음
- **rotation 이 진짜 방어**. history rewrite 는 부수적 흔적 제거

### Step 4 — 재스캔

```powershell
gitleaks detect --source . --no-banner --redact
```

exit 0 confirm.

### Step 5 — pre-commit hook 추가

`.pre-commit-config.yaml` 수정 (기존 ruff/mypy 다음에):

```yaml
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks
        name: gitleaks (CP239 — block secrets in commits)
```

활성화:
```powershell
cd C:\Users\user\lens
pre-commit install
pre-commit run gitleaks --all-files   # 1회 dry-run
```

### Step 6 — pre-commit hook 검증

```powershell
# 임시 가짜 secret 박은 파일 생성
echo "FAKE_AWS_KEY=AKIAIOSFODNN7EXAMPLE" > test_fake.env
git add test_fake.env
git commit -m "test: should be blocked"   # ← gitleaks hook 이 차단해야 함

# 차단 확인 후 즉시 삭제
git reset HEAD test_fake.env
rm test_fake.env
```

차단되면 `failed to find secret` 메시지가 안 나오고 `secret found` 메시지 + non-zero exit.

### Step 7 — CI workflow gitleaks job 추가

`.github/workflows/ci.yml` 끝에 신규 job 추가:

```yaml
  secrets:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0   # full history 필요

      - name: gitleaks (CP239 — secrets in history)
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}   # organization 면 license 필요. personal 은 무료
```

**Note**: `gitleaks-action@v2` 는 organization 계정에 라이센스 요구. personal repo (vmgfh878-art/lens) 는 무료. license env 없으면 그냥 동작.

대안 (라이센스 회피, 직접 binary):
```yaml
  secrets:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Install gitleaks
        run: |
          wget -q https://github.com/gitleaks/gitleaks/releases/download/v8.21.2/gitleaks_8.21.2_linux_x64.tar.gz
          tar -xzf gitleaks_8.21.2_linux_x64.tar.gz
          sudo mv gitleaks /usr/local/bin/
      - name: Detect secrets
        run: gitleaks detect --source . --no-banner --redact
```

### Step 8 — rotation runbook 작성

`docs/security/rotation_runbook.md`:

```markdown
# Secrets Rotation Runbook

> 사용 시점: gitleaks 발견 / 외부 노출 의심 / 정기 (6개월).
> 원칙: rotation 먼저, history rewrite 그 다음.

## 1. 사용 중 secrets 목록

| 변수명 | 사용처 | 발급 콘솔 | 영향 범위 |
|---|---|---|---|
| SUPABASE_URL | backend (CP236) | Supabase Project | API 호출 실패 → /health/ready degraded |
| SUPABASE_KEY (service_role) | backend | Supabase Project → Settings → API | 동일 |
| FRED_API_KEY | backend collector | https://fred.stlouisfed.org | macro indicator 수집 실패 |
| FMP_API_KEY | backend collector (optional) | https://site.financialmodelingprep.com | fundamentals 수집 |
| EODHD_API_KEY | backend collector (optional) | https://eodhd.com | EODHD 수집 |
| WANDB_API_KEY | ai 학습 스크립트 | https://wandb.ai/settings | 학습 시 W&B 업로드 실패 (학습 자체는 진행) |

## 2. Rotation 절차

### 공통
1. 새 key 발급
2. 로컬 `.env` 갱신 + Render dashboard env 갱신 + Vercel dashboard env 갱신
3. 로컬 backend + `/health/ready` + production smoke 통과 확인
4. 옛 key 폐기 (콘솔에서 revoke)
5. 로그 1주일 모니터링 (혹시 다른 곳에서 옛 key 호출 있는지)

### Supabase (가장 자주)
1. https://supabase.com/dashboard/project/<id>/settings/api
2. `service_role` 옆 "Reset" 클릭 → 새 key 복사
3. 로컬 + Render env 갱신
4. `curl /api/v1/health/ready` 확인
5. (anon key 는 클라이언트 노출이라 의도된 공개. 그래도 6개월 주기 rotate 권장)

### FRED / FMP / EODHD
1. 콘솔 → API keys → 옛 key 삭제 → 새 발급
2. 동일 절차

### W&B
1. https://wandb.ai/settings → API keys → revoke + new
2. 로컬 `~/.netrc` 갱신 또는 `wandb login --relogin`
3. 학습 스크립트 1회 dry-run 확인

## 3. History rewrite (옵션)
- gitleaks 발견 + rotation 끝낸 다음에만
- git filter-repo (권장) 또는 BFG
- force push 전 `git tag backup-pre-rewrite`
- force push 후 1주일 GitHub UI / Actions 캐시 모니터링

## 4. 사고 보고
- 외부 노출 발생 시 docs/security/incidents/YYYY-MM-DD_<topic>.md 작성
- 영향 범위 / rotation 시각 / history rewrite 여부 / 사후 조치 명시
```

### Step 9 — 보고서 + ADR

`docs/cp239_secrets_history_report.md`:

```markdown
# CP239 Secrets History 보고서

## gitleaks 스캔 결과
- 검사 범위: 전체 git history (commits N개)
- 발견: N건 (0이면 "Clean")
- 영향받은 secret: (있으면 목록)

## Rotation 실행 (발견 시)
- (각 provider 별 rotation 시각 + 영향 범위)

## History rewrite (실행 시)
- 도구: git filter-repo / BFG
- backup tag: backup-pre-rewrite-YYYYMMDD
- force push 시각

## 안전망 박음
- pre-commit gitleaks hook: active
- CI gitleaks job: active
- rotation runbook: docs/security/rotation_runbook.md

## 산출물
- docs/cp239_gitleaks_report.json
- .pre-commit-config.yaml diff
- .github/workflows/ci.yml diff
- docs/security/rotation_runbook.md
- docs/adr/0030_secrets_rotation_policy.md
```

---

## 5. 회귀 안전망

- Step 2 read-only → 회귀 0
- Step 3 rotation → 로컬 backend + production /health/ready 로 즉시 검증
- Step 5 pre-commit hook → Step 6 의 가짜 secret 차단 테스트로 hook 동작 확인
- Step 7 CI job → 첫 PR/push 에서 GREEN 확인

---

## 6. 성공 기준 (L8)

- `gitleaks detect` exit 0 (clean) 또는 발견 → rotate 완료 → 재검 0
- pre-commit gitleaks hook: 활성 + 가짜 secret commit 차단 확인
- CI gitleaks job: GREEN
- `docs/security/rotation_runbook.md` 1장 (provider 5개+ 절차)
- ADR-0030 작성

---

## 7. 인터페이스 보존 (L7)

- read-only 스캔: 영향 0
- rotation: 응답 schema 영향 0 (key 만 교체, endpoint 동일)
- 단, rotation 중 짧은 (1~2분) 다운타임 가능 — 사용자에게 사전 공지 권장

---

## 8. Lens 특화 (L9)

- Supabase service_role key 노출 = 최대 위험 (DB 전체 권한). anon key 노출 = 의도된 공개 (단 6개월 rotate)
- W&B key 노출 = 학습 로그 조작 가능. 영향 작지만 rotate
- FRED 무료 tier 라 rotate 영향 작음. 단 IP rate limit 영향 가능
- yfinance: key 없음 (rate limit 만 있음) — 스캔 대상 X

---

## 9. ADR-0030 작성 가이드

파일: `docs/adr/0030_secrets_rotation_policy.md`

```markdown
# ADR-0030: Secrets Detection + Rotation Policy

## Status
Accepted (2026-06-06)

## Context
git history 의 secret 노출 점검 없음. .gitignore 에 .env 박혀있지만 과거 commit에 실수로 들어갔을 가능성 0이 아님. v2 Supabase Auth 진입 전 rotation runbook 필요.

## Decision
1. **도구**: `gitleaks` (open-source, 룰셋 활발 유지, JWT/Supabase/AWS/Stripe/GitHub PAT 등 폭넓게 커버).
   - rejected: `trufflehog` (false positive 더 많음), `git-secrets` (AWS 중심).
2. **이중 게이트**: pre-commit hook (개발자 로컬) + CI job (push/PR). 둘 다 차단.
3. **Rotation 우선**: history rewrite 보다 rotation 이 진짜 방어. rewrite 는 흔적 제거 목적.
4. **Runbook**: docs/security/rotation_runbook.md 단일 진리. provider 별 절차 + 영향 범위.
5. **정기 rotation**: 6개월 (사용자 calendar reminder).
6. **사고 시**: 24시간 내 rotate, 72시간 내 history rewrite 검토.

## Consequences
- 개발자가 .env 실수 commit 시 pre-commit hook 차단 → 개발 흐름 약간 무거워짐
- 정기 rotation 운영 부담 (6개월) → 자동화 검토 (v2 시점)
- CI 에 gitleaks job 추가 → CI 시간 ~30초 증가

## References
- OWASP Top 10 A02
- gitleaks https://github.com/gitleaks/gitleaks
```

---

## 10. 자동 실행 적합도

| Step | 자동 | 사람 확인 |
|---|---|---|
| 1 | △ | scoop 명령은 자동, 직접 binary 다운로드는 수동 |
| 2 | ✅ | — |
| 3 | — | 분기 |
| 3a | ❌ | **provider 콘솔 수동** |
| 3b | ❌ | **dashboard 수동** |
| 3c | ❌ | **force push 결정 수동** |
| 4 | ✅ | — |
| 5 | ✅ | — |
| 6 | ✅ | — |
| 7 | ✅ | — |
| 8 | ✅ | — |
| 9 | ✅ | — |

→ **Step 3a/3b/3c (rotation) 은 절대 자동 X**. 발견 시 agent 가 멈춰서 사용자에게 alarm.

---

## 11. 종료 후 commit / 보고

### 권장 commit 분할

```
CP239 Step 1: gitleaks install + dev note
CP239 Step 2: gitleaks history scan (CLEAN | FOUND N)
# 발견 시:
CP239 Step 3a-c: rotation done + history rewrite (if applicable)
CP239 Step 4: post-rotation gitleaks rescan (CLEAN)
CP239 Step 5: pre-commit gitleaks hook
CP239 Step 6: pre-commit hook validation test
CP239 Step 7: CI workflow gitleaks job
CP239 Step 8: rotation runbook + provider 5 covered
CP239 report + ADR-0030
```

### 보고서
`docs/cp239_secrets_history_report.md`

### Runbook (신규)
`docs/security/rotation_runbook.md`

### ADR
`docs/adr/0030_secrets_rotation_policy.md`

---

**진입 조건**: 없음 (독립 CP). CP238 과 병렬 가능.
**다음 CP**: CP240 (HTTP 보안 헤더).
**리스크**: Step 3a~3c rotation 은 실제 서비스 영향. 발견 시 agent 멈춰서 사용자 alarm 후 진행.
