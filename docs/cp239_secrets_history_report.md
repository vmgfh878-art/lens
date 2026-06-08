# CP239 — Secrets History 보고서

작성: 2026-06-08. OWASP Top 10 A02. ADR-0030 동반.

## 0. 한 줄 요약

git history 1회 전체 scan + pre-commit/CI gate 시도 → **세 시도 모두 false
negative** 발견 + root cause (`[extend] useDefault = true` 누락) 진단 →
local 작동 확인 + CI false negative 미해결 → 사용자 명령 6 HARD STOP →
off-ramp ("`.gitignore` + 수동 spot-check + 정기 rotation" 으로 enforcement
대체). v2 재검토 commitment.

## 1. 핵심 컴포넌트 존재 체크리스트

CP239 는 정책 + 도구 시도 + off-ramp 결정. ML 컴포넌트 변경 0 (RevIN 등 N/A).
대신 본 CP 핵심 컴포넌트 체크:

- [x] `.gitleaks.toml` 신규 (`[extend] useDefault = true` + ML allowlist 좁힘)
- [x] gitleaks binary v8.21.2 → v8.30.1 (.venv\Scripts\gitleaks.exe, local-only)
- [x] `.github/workflows/ci.yml` 의 secrets job 추가 (gitleaks v8.30.1 + history 모드)
- [x] `docs/security/rotation_runbook.md` (6 provider 절차 + history rewrite + 사고 대응)
- [x] `docs/adr/0030_secrets_rotation_policy.md` (정책 + off-ramp 정직 기록 + v2 commitment)
- [x] `docs/cp239_gitleaks_report.json` (history scan baseline — 단 default rule 없는 silent-PASS 무효, 본 보고서 §3 참조)
- [x] pre-commit hook off-ramp (commit 5547a3e — local hook 제거)
- [x] backup-cp239-step6 tag 정리 (debug 종료)
- [x] cp239-cigate-test scratch branch 정리 (다른 세션 도움)
- [x] 운영 코드 / ML 모델 / 응답 schema 0 수정

## 2. 새 테스트 결과

CP239 자체는 새 unit test 추가 없음 (의존성 도구 시도 + 정책 박음). 회귀
안전망 재실행:

```
pytest backend/tests -k "snapshot or characterization or cp223" -q
21 passed, 110 deselected
```

- characterization snapshot 9 endpoint ✅
- drift simulation 11 케이스 ✅
- 기타 cp223 매치 1 ✅

```
npm run test:unit (vitest)
8 files / 166 tests passed
```

## 3. 진행 — 시도 + 결과 정직 기록

### 3.1 Step 1-2: gitleaks install + history scan

- `gitleaks v8.21.2` direct binary install → `.venv\Scripts\gitleaks.exe`
- `gitleaks detect --source .` 전체 history scan → **"192 commits scanned,
  no leaks found"** 결과 (clean 판정)
- 단 본 결과는 **무효** (Step 7 의 root cause 발견 — `[extend] useDefault =
  true` 누락으로 default rule 미적용 silent-PASS)

### 3.2 Step 3-4: false positive 1건 + allowlist + 재스캔

- 1 finding (false positive): `metricKey: "lower_breach_fold2"` (ML WFLOCK
  fold metric identifier, `generic-api-key` entropy 오탐)
- `.gitleaks.toml` allowlist 박음 (paths + regexes)
- 재스캔 → 0 leaks (역시 silent-PASS 무효)

### 3.3 Step 5-6: pre-commit hook 시도 + off-ramp

| 시도 | 결과 |
|---|---|
| v8.21.2 hook entry `gitleaks protect --staged` | deprecated, silent PASS. commit 9032bfe (가짜 AKIA EXAMPLE) 통과 |
| v8.30.1 hook entry `gitleaks git --pre-commit --redact --staged --verbose` | cache binary 직접 실행 시 "0 commits scanned, no leaks". commit d0e2fb0 (가짜 3 secret + rev bump) 통과 |
| local hook `entry: ./.venv/Scripts/gitleaks.exe detect --no-git --source . --redact --no-banner` | 1m53s 검사 + "files were modified by this hook" + "no leaks found" Failed. backend/app/leaked_config.py 의 secret 못 잡음 |

→ off-ramp (commit 5547a3e): `.pre-commit-config.yaml` 의 gitleaks block 제거.

### 3.4 Step 7: CI gate 추가 + root cause 발견 + false negative 확정

CI workflow `secrets` job 추가 (commit f909f24):
```yaml
- gitleaks detect --redact --no-banner   # history 모드, default
```

검증 (scratch branch `cp239-cigate-test` + 16f6652 가짜 ghp_PAT push):
- CI run 27125419652 secrets job: **success** (못 잡음)
- allowlist 좁힘 (commit 149098d, regexes + `.gitleaks.toml$` path 제거):
  여전히 success

**root cause 발견 (사용자 가설 정확)**:
- `.gitleaks.toml` 의 `[allowlist]` 만 박고 `[extend] useDefault = true` 없으면
  default secret rules **자체가 미적용**
- 사용자 명령 [extend] useDefault 추가 (commit 56c77ae):
  ```toml
  [extend]
  useDefault = true
  ```
- 로컬 검증 (leaked_config.py 임시 + `gitleaks detect --no-git --source .`):
  **leaks found 296** (default rules 활성)
- 로컬 history 검사 (`gitleaks detect --redact --no-banner`): **1 leak
  (github-pat, 16f6652)** — 의도된 가짜 정확 검출 ✅
- CI 재실행 (main push 27127329020): secrets job **success** (또 못 잡음) ❌

→ **사용자 명령 6 HARD STOP**: "step 2 통과 + step 4 실패" 시 진짜 off-ramp.

CI false negative 원인 미진단 (사용자 명령 "더 시도 X"). 추정:
- Linux ubuntu binary 의 `.gitleaks.toml` 자동 로드 동작 차이
- fetch-depth: 0 의 어떤 edge case
- v8.30.1 의 OS 별 동작 차이

### 3.5 Step 8-9: off-ramp 정직 문서화

- `docs/security/rotation_runbook.md`: 6 provider 절차 (Supabase / FRED /
  FMP / EODHD / W&B / anon) + 수동 spot-check + history rewrite + 사고 대응
- `docs/adr/0030_secrets_rotation_policy.md`: 정책 + off-ramp 정직 기록 + v2
  commitment
- backup tag + scratch branch 정리

## 4. 기존 회귀 통과 건수

### 4.1 회귀 안전망

| 안전망 | 결과 |
|---|---|
| CP223 characterization snapshot | ✅ 9 passed |
| CP237.5 drift simulation | ✅ 11 passed |
| CP230 frontend smoke | ✅ 8 files / 166 passed |
| CP238 pip-audit allowlist | ✅ exit 0 |
| CP238 audit-ci allowlist | ✅ exit 0 |
| CP239 gitleaks local | ⚠️ working (default rules 활성) but CI false negative |

### 4.2 backend pytest 전체 (참고)
118 passed, 11 failed (CP237.5 보고서 §4.2 의 pre-existing 동일).

## 5. 진행 중 발견 + 정직 한계 6 개

1. **`[extend] useDefault = true` 누락** (root cause): config file 만 박으면
   default rules 미적용 silent-PASS. 사용자 가설 정확.

2. **Windows pre-commit hook 3 시도 모두 차단 X**:
   - v8.21.2 `protect` deprecated
   - v8.30.1 `git --pre-commit --staged` "0 commits scanned"
   - local `detect --no-git --source .` allowlist 흡수 mystery

3. **`.gitleaks.toml` 자동 로드 + `--source .` 시 backend/app/leaked_config.py
   의 secret 도 흡수 mystery** (local 환경 한정. v2 재검토).

4. **CI Linux ubuntu false negative**: useDefault 박힌 후에도 history 의
   가짜 PAT 못 잡음. 원인 미진단.

5. **다른 세션 동시 진행**: 308c7ec (test_cigate.txt 제거) + 06dc6ee (docs)
   + ea71b68 (html) 다른 세션이 origin/main 갱신. 다른 세션이 cp239-cigate-test
   branch 자체 삭제 (cleanup). cooperation, conflict 0.

6. **main history 의 가짜 PAT 영구 잔존**: 16f6652 (cp239-cigate-test 의 의도된
   가짜 ghp_PAT 박은 commit) 가 main 의 ancestor. 다른 세션의 308c7ec 가 file 만
   삭제, history rewrite 없음. **가짜 (random alphabet sequential, real PAT
   아님) 라 rotation 면제 acknowledged**. history rewrite 진행 안 함 (force
   push 위험 + 진짜 secret 아님).

## 6. 산출물

### 신규
- `.gitleaks.toml` (`[extend] useDefault = true` + ML allowlist 좁힘)
- `docs/security/rotation_runbook.md` (6 provider 절차 + history rewrite + 사고 대응)
- `docs/adr/0030_secrets_rotation_policy.md` (정책 + off-ramp 정직 기록 + v2 commitment)
- `docs/cp239_gitleaks_report.json` (silent-PASS 무효 baseline, §3.1 명시)
- `docs/cp239_secrets_history_report.md` (본 보고서)

### 수정
- `.github/workflows/ci.yml` (secrets job 추가, false negative confirmed)
- `.pre-commit-config.yaml` (gitleaks block off-ramp 제거)

### 정리
- `backup-cp239-step6` tag 삭제
- `cp239-cigate-test` scratch branch 삭제 (local + origin)

### 운영 코드 / ML 모델 / 응답 schema 0 수정

## 7. commit 이력 (CP239 만, 14 commit)

```
2307c11 CP239 Step 2-4: gitleaks history scan (CLEAN after false-positive allowlist)
acbb46d CP239 Step 5: pre-commit gitleaks hook
9032bfe test: should be blocked by gitleaks       [v8.21.2 protect deprecated 검증, reset 됨]
e9f5c70 test: ghp PAT should be blocked            [v8.30.1 hook 검증, reset 됨]
d0e2fb0 test: 3 secret patterns should be blocked  [verify-first, reset 됨]
5547a3e CP239 Step 5-6 off-ramp: drop local gitleaks hook (CI gate is enforcement)
f909f24 CP239 Step 7: CI workflow secrets job (gitleaks history gate)
2791c63 test: ci-gate verification (delete after)  [main 사고 commit, reset 됨]
16f6652 test: ci-gate verification (delete after)  [cp239-cigate-test 의 가짜 PAT — main ancestor 영구 잔존]
149098d CP239 Step 7 검증: .gitleaks.toml allowlist 좁힘
56c77ae CP239 Step 7 fix: .gitleaks.toml 에 [extend] useDefault = true 추가
308c7ec chore: ci-gate 검증 테스트 잔재(test_cigate.txt) 제거 [다른 세션]
06dc6ee docs: 종합 보고서 데모 화면 개선 + README 링크 단일화 [다른 세션]
<본 commit> CP239 Step 8-9 closure: runbook + ADR-0030 + report
```

## 8. v2 재검토 commitment (ADR-0030 §6)

- Supabase Auth + production 본격화 트랙 진입 시 본 CP239 의 §3a/3b/3c (off-ramp)
  재시도
- gitleaks 대체 도구 검토: `trufflehog`, `gitleaks-action@v2`, 직접 shallow
  grep CI script
- ML training metric identifier false positive 흡수 mystery 도 재진단

## 9. 다음 CP

CP240 (HTTP 보안 헤더 + CSP). 진입 조건 (CP223 snapshot + CP230 FE smoke GREEN)
충족.
