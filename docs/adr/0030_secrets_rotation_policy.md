# ADR-0030: Secrets Detection + Rotation Policy

Status: Accepted (with v2 re-evaluation commitment)
Date: 2026-06-08
CP: CP239 (OWASP A02)

## Context

git history 의 secret 노출 점검을 한 번도 한 적 없음. `.gitignore` 에 `.env`
박혀있지만 과거 commit 에 실수로 들어갔을 가능성 0 이 아님. v2 Supabase Auth
진입 전 rotation runbook + 자동 detection 필요.

## Decision

### 1. 도구 선정 (의도)

- **`gitleaks`** (open-source, 룰셋 활발 유지, JWT/Supabase/AWS/Stripe/GitHub PAT
  등 폭넓게 커버)
- 기각 대안:
  - `trufflehog`: false positive 더 많음
  - `git-secrets`: AWS 중심

### 2. 이중 게이트 (의도)

- **pre-commit hook** (개발자 로컬, 즉시 차단)
- **CI job** (push/PR, 머지 차단)
- 둘 다 active

### 3. 실측 결과 — gitleaks 환경 신뢰 불가 (off-ramp)

CP239 Step 6-7 검증 결과 **세 가지 정직 한계** 발견:

#### 3a. Windows pre-commit hook unreliable

| 시도 | 결과 |
|---|---|
| rev v8.21.2 entry `gitleaks protect --staged` | `protect` 가 v8.18+ 에서 deprecated → silent PASS |
| rev v8.30.1 entry `gitleaks git --pre-commit --staged --verbose` | "0 commits scanned, no leaks found" — `--staged` 가 staged content 검사 안 함 |
| local hook `./.venv/Scripts/gitleaks.exe detect --no-git --source . --redact --no-banner` | 1m53s 검사 + "files were modified by this hook" + "no leaks found" — `--source .` (전체 dir) + `.gitleaks.toml` 자동 로드의 mystery 흡수 동작 |

→ **off-ramp**: `.pre-commit-config.yaml` 의 gitleaks hook 제거 (commit 5547a3e).

#### 3b. `.gitleaks.toml` 의 `[extend] useDefault` 필수 (root cause 발견)

CP239 Step 7 검증 mystery 의 진짜 원인:

> gitleaks 의 default secret rules (`github-pat` / `aws-access-token` /
> `slack-bot-token` / ...) 가 자동 적용되는 게 아니다. config file 이 존재하면
> `[extend] useDefault = true` 없이는 default rule set 자체가 미적용 → **어떤
> secret pattern 도 룰에 없어 silent-PASS**.

CP239 Step 2 의 "192 commits scanned, no leaks found" 결과는 사실 default
rule 없는 silent-PASS 라 **무효**.

수정 (commit 56c77ae):
```toml
[extend]
useDefault = true
```

수정 후 local 검증:
- `gitleaks detect --no-git --source .` (leaked_config.py 임시 두고):
  leaks found 296 (default rules 활성)
- `gitleaks detect --redact --no-banner` (history): **1 leak (github-pat,
  cp239-cigate-test 의 16f6652 의 의도적 가짜 test_cigate.txt)** — 의도된 가짜만
  정확히 검출

→ **local 환경에서 gitleaks 정상 작동 확인**.

#### 3c. CI Linux ubuntu binary false negative

`useDefault` 박힌 `.gitleaks.toml` push 후 CI 의 `gitleaks detect` (동일 명령):
- secrets job conclusion = **success** (못 잡음)
- 16f6652 의 가짜 ghp_PAT 가 main history 의 ancestor 인 게 확인 됐는데도

원인 미진단 (사용자 명령 "더 시도 X"). 추정:
- Linux ubuntu binary 의 `.gitleaks.toml` 자동 로드 path 차이
- fetch-depth: 0 의 어떤 edge case
- gitleaks v8.30.1 의 OS 별 동작 차이

→ **CI gate 도 신뢰 불가**. 

#### 결론 (HARD STOP, 사용자 명령 6 발동)

"useDefault 넣어도 step 2 (local) 또는 step 4 (CI) 에서 여전히 안 잡으면 →
진짜 off-ramp". 우리 케이스: **step 2 통과 + step 4 실패**.

→ **gitleaks 자동 검출 enforcement 미작동 acknowledged**. 

### 4. 진짜 enforcement = `.gitignore` + 수동 spot-check + 정기 rotation

대체 안전망 (`docs/security/rotation_runbook.md` §0/§4):

| 안전망 | 상태 |
|---|---|
| `.gitignore` 의 `.env` / `*.env` / `.env.local` | ✅ 1차 방어 (env file commit 차단) |
| `gitleaks` pre-commit / CI | ❌ off-ramp (false negative confirmed) |
| 수동 spot-check (`git ls-files`, `git log -p` grep) | ✅ 정기 1개월 + 신규 endpoint 시 |
| 정기 rotation 6개월 | ✅ 캘린더 박힘 |

### 5. Rotation 우선

History rewrite 보다 rotation 이 진짜 방어. rewrite 는 흔적 제거 목적.
`docs/security/rotation_runbook.md` 가 단일 진리.

### 6. v2 재검토 (commitment)

다음 트랙 진입 시 본 ADR 의 §3a/3b/3c 재시도:
- Supabase Auth 도입 = 인증 도입 = attack surface 증가
- 그 시점에 gitleaks 대체 도구 시도:
  - `trufflehog` (다른 룰 엔진)
  - `gitleaks-action@v2` (GitHub-hosted runner 의 다른 binary)
  - 또는 직접 작성한 shallow grep CI script (정직히 단순)
- ML training metric identifier false positive 흡수 mystery 도 재진단

## Consequences

### 장점
- `.gitignore` 의 `.env` 차단 + 수동 spot-check + 정기 rotation 으로 1차 방어 박힘
- 정직성 lock: 자동 detection 미작동 명시 + v2 commitment
- rotation_runbook.md 1장 (6 provider 절차 + history rewrite 절차 + 사고 대응)

### 단점 / Trade-off
- 개발자가 `.env` 외 다른 경로에 secret 박을 가능성 (auto detection 미작동)
- 정기 6개월 rotation 운영 부담 — 자동화 검토 (v2 시점)
- v1 운영 진입 시점에 OWASP A02 자동 방어 약함 acknowledged

### Mitigation
- `.gitignore` 가 충분한 1차 방어 (가장 흔한 실수 차단)
- Lens v1 의 attack surface 작음 (인증 0, 공개 read-only API)
- 정기 rotation + 수동 spot-check 가 자동 detection 대체

## Backup tag 정리

CP239 디버그 중 박았던 `backup-cp239-step6` tag (9032bfe AKIA test commit)
는 CP239 closing 시 삭제 (정직성: 더 이상 디버그 안 함).

cp239-cigate-test scratch branch 도 삭제 (다른 세션이 정리 도움 +
push delete).

main history 의 16f6652 (test_cigate.txt 의 가짜 ghp_PAT) 는 다른 세션의
test_cigate.txt 제거 commit (308c7ec) 후에도 ancestor 로 잔존 — 그러나
이는 의도된 가짜 (random alphabet sequential, real PAT 아님) acknowledged.
history rewrite 진행하지 않음 (irreversible, force push 위험 + 진짜
secret 아님).

## References

- OWASP Top 10 A02 (https://owasp.org/Top10/A02_2021-Cryptographic_Failures/)
- gitleaks (https://github.com/gitleaks/gitleaks)
- 본 트랙 보고서: `docs/cp239_secrets_history_report.md`
- 사용자 가이드: `docs/security/rotation_runbook.md`
- false positive allowlist: `.gitleaks.toml`
