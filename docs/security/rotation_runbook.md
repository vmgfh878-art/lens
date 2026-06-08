# Secrets Rotation Runbook

> 작성: 2026-06-08 (CP239). 사용 시점: gitleaks 발견 / 외부 노출 의심 / 정기
> 6개월 / v2 (Auth) 진입 시점. 원칙: **rotation 이 진짜 방어**, history rewrite
> 는 부수적 흔적 제거.

---

## 0. CP239 안전망 현황 (정직 명시)

| 안전망 | 상태 |
|---|---|
| `.gitignore` 의 `.env` / `*.env` / `.env.local` | ✅ 박혀있음 (1차 방어) |
| gitleaks pre-commit hook (Windows) | ❌ off-ramp (v8.21.2 protect deprecated / v8.30.1 git --pre-commit staged 검사 X / local detect 의 `.gitleaks.toml` 자동 로드 동작 mystery) |
| gitleaks CI gate (Ubuntu) | ⚠️ **false negative** — local `gitleaks detect` 는 가짜 PAT 정확히 1 leak 검출 (history 모드), 그러나 동일 명령이 CI Linux ubuntu binary 에서 secrets job conclusion = success (못 잡음). config 로딩 / fetch-depth / OS binary 차이 추정. **v2 재검토** |
| 정기 rotation (6개월) | ✅ 본 runbook 으로 절차 박음 |
| 사용자 수동 spot-check | ✅ 사용 중 secrets 별 grep 권장 (아래 §3) |

→ **현 시점 진짜 enforcement = `.gitignore` + 수동 spot-check + 정기 rotation**.
gitleaks 자동 검출은 미작동 (정직). v2 (Supabase Auth + production 본격화) 시
gitleaks-action 또는 trufflehog 등 대체 도구로 재시도.

---

## 1. 사용 중 secrets 목록

| 변수명 | 사용처 | 발급 콘솔 | 영향 범위 (노출 시) |
|---|---|---|---|
| `SUPABASE_URL` | backend (CP236) | Supabase Project Settings | API 호출 정보 노출 (public 도 OK 한 URL) |
| `SUPABASE_KEY` (service_role) | backend | Supabase Project → Settings → API | **DB 전체 권한** — 최대 위험. 즉시 rotation 필수 |
| `FRED_API_KEY` | backend collector | https://fred.stlouisfed.org/docs/api/api_key.html | macro indicator 수집 실패 (rate limit). 영향 작음 |
| `FMP_API_KEY` | backend collector (optional) | https://site.financialmodelingprep.com (Dashboard) | fundamentals 수집 실패 |
| `EODHD_API_KEY` | backend collector (optional) | https://eodhd.com (Dashboard) | EODHD 수집 실패 |
| `WANDB_API_KEY` | ai 학습 스크립트 | https://wandb.ai/settings | 학습 로그 조작 가능. 학습 자체는 진행 |

**anon key (Supabase, 클라이언트 노출 의도된 공개)**: 노출 자체 정상. 6개월 rotation
권장.

---

## 2. Rotation 절차 (공통)

1. 새 key 발급 (provider 콘솔)
2. 로컬 `backend/.env` 갱신
3. Render dashboard → Services → lens-backend → Environment 에서 env var 교체
   → Save → 자동 redeploy
4. Vercel dashboard → lens project → Settings → Environment Variables 에서
   교체 → Redeploy 트리거
5. 로컬에서 backend 띄워서 `/health/ready` 정상 확인:
   ```powershell
   cd C:\Users\user\lens\backend
   .\.venv\Scripts\uvicorn.exe app.main:app --port 8000
   # 다른 PowerShell 창에서:
   curl http://127.0.0.1:8000/api/v1/health/ready
   ```
6. production smoke 확인 (vercel.app + render backend URL)
7. **옛 key 폐기** (콘솔에서 revoke). 새 key 동작 확인 후 즉시 폐기 필수
8. 로그 1주일 모니터링 — 혹시 다른 곳에서 옛 key 호출 흔적

---

## 3. Provider 별 rotation 절차

### 3.1 Supabase service_role key (가장 자주, 가장 위험)

1. https://supabase.com/dashboard/project/<your-project-id>/settings/api
2. `service_role` key 옆 **"Reset"** 또는 **"Regenerate"** 클릭
3. 새 key 복사 (한 번만 표시됨)
4. §2 공통 절차로 로컬 + Render env 갱신
5. `curl /api/v1/health/ready` 정상 확인
6. 옛 key 자동 폐기 (Supabase 가 reset 시 옛 key 즉시 무효)

### 3.2 Supabase anon key (선택, 6개월)

1. https://supabase.com/dashboard/project/<id>/settings/api
2. `anon` key 옆 "Reset"
3. Vercel dashboard 의 `NEXT_PUBLIC_SUPABASE_ANON_KEY` 갱신
4. 모든 사용자가 새 client 로드 후 옛 key 무효

### 3.3 FRED API Key

1. https://fred.stlouisfed.org → My Account → API Keys
2. 옛 key 삭제 → 새 key 발급
3. §2 공통 절차 (collector 만 영향, /health/ready 안 거침 — collector 스크립트
   직접 실행해서 확인)

### 3.4 FMP / EODHD API Key

1. 콘솔 → API Keys → 옛 key 삭제 → 새 발급
2. §2 공통 절차

### 3.5 W&B API Key

1. https://wandb.ai/authorize 또는 https://wandb.ai/settings → API keys
2. revoke + new
3. 로컬:
   ```powershell
   .\.venv\Scripts\wandb.exe login --relogin
   ```
4. 학습 스크립트 1회 dry-run 으로 wandb upload 확인

---

## 4. 수동 spot-check (gitleaks 미작동 대체)

gitleaks 가 enforcement 못 하는 상황의 fallback. 정기 (1개월) 또는 신규
endpoint / collector 추가 시:

```powershell
cd C:\Users\user\lens
# .env 패턴 추적 확인 (.gitignore 효과 검증)
git ls-files | Select-String "\.env$|\.env\." 2>&1
# → 비어있어야 정상 (.env 류 추적 안 됨)

# 신규 commit 의 대표 secret pattern grep
git log -p -3 | Select-String "ghp_|AKIA|xoxb-|sk-[a-zA-Z0-9]|API_KEY\s*=" -CaseSensitive
# → matched line 박혀있으면 정직히 확인 (가짜인지 진짜인지)
```

PowerShell 사용 (Windows). bash 환경에선 `grep -E`.

---

## 5. History rewrite (옵션, 신중)

`gitleaks` (또는 manual grep) 으로 진짜 secret 발견 시:

**원칙**: rotation 먼저, history rewrite 그 다음. GitHub cached views / PR diff
/ fork 가 secret 보존 가능 → **rotation 이 진짜 방어**.

### 5.1 git filter-repo (권장)

```powershell
cd C:\Users\user\lens
git tag backup-pre-rewrite-$(Get-Date -Format yyyyMMdd)
pip install git-filter-repo
git filter-repo --invert-paths --path backend/scripts/leaked.py   # 예시
```

### 5.2 BFG Repo-Cleaner

```powershell
# bfg.jar 다운로드 후
java -jar bfg.jar --replace-text passwords.txt   # passwords.txt 에 매칭 패턴
```

### 5.3 Force push

```powershell
git push --force-with-lease origin main
```

**경고**:
- GitHub 의 cached views, PR diff, raw URL 은 force push 후에도 일정 시간
  secret 노출 가능
- 협업자 (1인이라 없지만) 가 fork 했으면 그쪽에 남음
- 1주일 GitHub Actions / Insights / Network graph 모니터링

---

## 6. 사고 대응 절차

외부 노출 또는 의심 발생 시:

1. **즉시 rotation** (24시간 내) — §3 의 해당 provider 절차
2. 영향 범위 평가 — 옛 key 로 호출된 API / DB 접근 / 학습 로그 등
3. 외부 노출 흔적 정리 (GitHub force push, third-party log 등 — 72시간 내)
4. `docs/security/incidents/YYYY-MM-DD_<topic>.md` 작성:
   - 발견 시각 / 발견 경로 (gitleaks / 수동 / 외부 보고 / log anomaly)
   - 노출 secret + 영향 범위
   - rotation 시각 + 절차
   - history rewrite 여부 + 절차
   - 사후 조치 (모니터링 / 추가 안전망)

---

## 7. 정기 rotation 캘린더

- Supabase service_role + anon: 6개월 (다음 2026-12-08)
- FRED / FMP / EODHD: 6개월 (동일)
- W&B: 6개월 (동일)
- v2 (Supabase Auth + production 본격화): 동시 rotation 권장 + gitleaks 재시도

---

## References

- ADR-0030 (`docs/adr/0030_secrets_rotation_policy.md`)
- CP239 보고서 (`docs/cp239_secrets_history_report.md`)
- OWASP A02 Cryptographic Failures
- gitleaks https://github.com/gitleaks/gitleaks
