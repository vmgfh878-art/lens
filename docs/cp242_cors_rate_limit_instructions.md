# CP242 지시서 — CORS Allowlist 점검 + Rate Limit 검토 + 보안 트랙 종료 (OWASP A04/A05)

> 작성: 2026-06-06. 트랙: Lens 보안 트랙 마지막. ADR-0033 동반.
> 사용자 환경: Windows PowerShell. 한국어 보고서.

---

## 0. 한 줄 목표

`CorsConfig` (CP235) 가 production 에서 `*` 안 쓰는지 점검. preview 도메인 정규식 적용. (선택) `slowapi` rate limit 도입 결정. CP238~242 보안 트랙 종합 보고서 + ADR-0033.

---

## 1. 진단

| 항목 | 현재 |
|---|---|
| CORS 정리 | CP235 `CorsConfig` (Pydantic Settings) — `BACKEND_CORS_ORIGINS` / `BACKEND_CORS_ORIGIN_REGEX` env |
| production `BACKEND_CORS_ORIGINS` 실제 값 | **미확인** (Render dashboard 직접 봐야) |
| `allow_credentials` 설정 | 미확인 (`main.py` 의 `CORSMiddleware` 인자) |
| Vercel preview 도메인 처리 | 미확인 (regex 사용 여부) |
| Rate limit | 없음 (`slowapi` 미설치) |
| `/health/live` rate limit 예외 | 해당 없음 (rate limit 자체 없음) |

**OWASP**: A04 Insecure Design + A05 Misconfiguration.

---

## 2. 변경 내용

### 2.1 CORS (필수)
- `CorsConfig` 코드에서 `*` 사용 가능성 확인
- Render dashboard 에서 production `BACKEND_CORS_ORIGINS` 실제 값 확인 → `*` 있으면 즉시 교체
- `allow_credentials=True` 인 경우 `*` 절대 금지 (Spec 위반) 확인
- Vercel preview 도메인 정규식 적용 (`BACKEND_CORS_ORIGIN_REGEX`)

### 2.2 Rate Limit (선택)
- `slowapi` 도입 여부 결정 (v1 도입 권장, 가볍게)
- 도입 시: `/api/v1/predictions/*` 분당 60회, `/health/live` 무제한, search 분당 30회

### 2.3 트랙 종료
- `docs/cp242_security_track_summary.md` 작성 (CP238~242 종합)
- ADR-0033

---

## 3. Step 분할

| Step | 내용 | 위험 | 시간 | 자동/수동 |
|---|---|---|---|---|
| 1 | `backend/app/config/settings.py` `CorsConfig` 코드 read → `*` 사용 가능성 확인 | 매우낮음 | 10분 | 자동 |
| 2 | **Render dashboard** 에서 `BACKEND_CORS_ORIGINS` env 실제 값 확인 → `*` 있으면 명시 origin 으로 교체 | 낮음 | 10분 | **수동** |
| 3 | `main.py` `CORSMiddleware` 인자 점검: `allow_credentials=True` + `allow_origins=["*"]` 조합 금지 (Spec 위반) | 매우낮음 | 10분 | 자동 |
| 4 | Vercel preview 도메인 정규식 (`BACKEND_CORS_ORIGIN_REGEX`) 적용: `^https:\/\/lens(-[a-z0-9-]+)?\.vercel\.app$` | 낮음 | 15분 | 자동 |
| 5 | **Rate limit 도입 여부 결정** (v1: 가볍게 도입 권장 / v2: 본격) | — | 15분 | **사용자 결정** |
| 6 | (도입 시) `slowapi` 설치 + Limiter 적용 + 핵심 endpoint `@limiter.limit("60/minute")` | 중간 | 1h | 자동 |
| 7 | (도입 시) `/health/live` 예외 + 429 응답 schema 통일 + pytest | 낮음 | 30분 | 자동 |
| 8 | `docs/cp242_security_track_summary.md` 작성 (CP238~242 종합) | 매우낮음 | 30분 | 자동 |
| 9 | ADR-0033 작성 (CORS 정책 + rate limit 결정 + 보안 트랙 회고) | 매우낮음 | 30분 | 자동 |

---

## 4. 각 Step 정확한 명령 / 코드

### Step 1 — CorsConfig 코드 점검

```powershell
cd C:\Users\user\lens
grep -n "CorsConfig\|allow_origins\|allow_credentials\|cors" backend/app/config/settings.py backend/app/main.py
```

확인할 것:
1. `CorsConfig.origins` 가 `["*"]` 으로 fallback 되는 path 있는지 (예: env 없을 때)
2. `_DEFAULT_CORS_ORIGINS` 값
3. `BACKEND_CORS_ORIGIN_REGEX` 변환 로직

만약 fallback 으로 `["*"]` 있다면 → **production 에서 절대 fallback 되지 않게 검증 로직 추가**:

```python
# backend/app/config/settings.py - CorsConfig 안
def model_post_init(self, __context) -> None:
    if self.environment == "production" and "*" in self.origins:
        raise ValueError(
            "BACKEND_CORS_ORIGINS='*' is forbidden in production. "
            "Set explicit origins."
        )
```

(`environment` 필드가 별도 Settings 에 있으면 그쪽으로.)

### Step 2 — Render dashboard 확인 (수동)

사용자가 직접:
1. https://dashboard.render.com → lens-backend service
2. Environment 탭
3. `BACKEND_CORS_ORIGINS` 값 확인

기대값:
```
https://lens-ten-delta.vercel.app
```

또는 (preview 포함):
```
https://lens-ten-delta.vercel.app,https://lens-preview.vercel.app
```

`*` 있으면 즉시 위 값으로 교체 → Save → 자동 redeploy.

### Step 3 — main.py CORSMiddleware 점검

```powershell
grep -n -A 8 "CORSMiddleware" backend/app/main.py
```

기대 패턴:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors.origins,
    allow_origin_regex=_cors.origin_regex,
    allow_credentials=True,   # 또는 False
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```

규칙:
- `allow_credentials=True` 이면 `allow_origins=["*"]` 금지 (Spec 위반, 브라우저가 거부)
- preview 도메인은 `allow_origin_regex` 로 처리

만약 둘 다 안전 confirm 되면 변경 없음. ADR 에 명시.

### Step 4 — Vercel preview regex

`backend/app/config/settings.py` `CorsConfig` 에 regex 필드 추가:

```python
class CorsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    raw_origins: str = Field(
        default=_DEFAULT_CORS_ORIGINS,
        alias="BACKEND_CORS_ORIGINS",
    )
    origin_regex: str | None = Field(
        default=r"^https://lens(-[a-z0-9-]+)?\.vercel\.app$",
        alias="BACKEND_CORS_ORIGIN_REGEX",
    )

    @property
    def origins(self) -> list[str]:
        return [s.strip() for s in self.raw_origins.split(",") if s.strip()]
```

`main.py` 에서 사용:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors.origins,
    allow_origin_regex=_cors.origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```

검증:
```powershell
# 로컬 backend 띄우고
curl -H "Origin: https://lens-preview-abc.vercel.app" -I http://127.0.0.1:8000/api/v1/health/live
# 응답에 access-control-allow-origin: https://lens-preview-abc.vercel.app
```

### Step 5 — Rate limit 결정

**판단 보조 표**:

| 옵션 | 장점 | 단점 |
|---|---|---|
| v1 도입 X | 가벼움, 사용자 작아 DoS 위험 낮음 | "보안 검토 완료" 신호 약함 |
| v1 가볍게 도입 (분당 60회) | 포트폴리오 신호 + DoS baseline | slowapi 의존성 추가 (가볍지만) |
| v1 본격 도입 (path 별 차등) | 시니어급 | 과투자 (사용자 본인 + 평가자) |

**권장**: **v1 가볍게 도입**. slowapi 가볍고, Render free tier 한 사용자 폭주 방지에 실용 가치. v2 Supabase Auth 진입 시 login brute force 방어 토대.

→ Step 6, 7 진행.

만약 보류 결정: Step 6, 7 skip → Step 8, 9 로 점프 (ADR 에 "v1 보류, v2 도입" 명시).

### Step 6 — slowapi 도입

`backend/requirements.txt` 에 추가:
```
slowapi==0.1.9
```

설치:
```powershell
cd C:\Users\user\lens
pip install slowapi==0.1.9
```

`backend/app/core/rate_limit.py` (신규):

```python
"""CP242 — Rate limiting (IP 기반, in-memory).

v1: 가벼운 baseline. 분당 60회 (사용자 본인 + 평가자 충분).
v2: Supabase Auth 진입 시 login brute force 방어 본격화.

스토리지: in-memory (단일 인스턴스라 충분). 다중 인스턴스 시 Redis 로 옮김 (v2).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# X-Forwarded-For 헤더 신뢰 — Render proxy 가 set.
# 만약 proxy 신뢰 안 하면 raw remote_addr 만 사용.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],   # 명시적 endpoint 만 제한
)
```

`backend/app/main.py` 에 add:

```python
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.rate_limit import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

핵심 endpoint 에 데코레이터:

```python
from app.core.rate_limit import limiter
from fastapi import Request

@router.get("/line/{ticker}")
@limiter.limit("60/minute")
def get_line(request: Request, ticker: TickerStr, ...):
    ...

@router.get("/band/1d/{ticker}")
@limiter.limit("60/minute")
def get_band_1d(request: Request, ticker: TickerStr, ...):
    ...

@router.get("/stocks")
@limiter.limit("30/minute")   # search 는 더 strict
def search_stocks(request: Request, ...):
    ...
```

(`Request` 인자 추가 필수 — slowapi 가 request.state 읽어야 함.)

### Step 7 — /health/live 예외 + 429 schema + pytest

`/health/live` 는 UptimeRobot ping 대상이라 rate limit 적용 시 false alarm 위험.

```python
@router.get("/health/live")
def health_live():
    # rate limit decorator 없음 → 무제한
    return {"status": "ok"}
```

429 응답 schema 통일 — main.py 에서 `RateLimitExceeded` 커스텀 핸들러:

```python
from slowapi.errors import RateLimitExceeded

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please retry after a moment.",
            },
            "meta": {"request_id": getattr(request.state, "request_id", None)},
        },
        headers={"Retry-After": "60"},
    )
```

pytest:
```python
# backend/tests/test_cp242_rate_limit.py
def test_rate_limit_triggers():
    """61번 호출 시 429 발생."""
    for i in range(60):
        r = client.get("/api/v1/predictions/line/AAPL")
        assert r.status_code in (200, 404)
    r = client.get("/api/v1/predictions/line/AAPL")
    assert r.status_code == 429


def test_health_live_no_rate_limit():
    """/health/live 는 무제한."""
    for _ in range(100):
        r = client.get("/api/v1/health/live")
        assert r.status_code == 200
```

(주의: slowapi 가 TestClient 환경에서 동작 안 할 수 있음 → `pytest.mark.skipif` 또는 `monkeypatch` 로 limiter 재설정.)

### Step 8 — 보안 트랙 종합 보고서

`docs/cp242_security_track_summary.md`:

```markdown
# CP238~CP242 Lens 보안 트랙 종합 보고서

> 작성: 2026-06-06. 트랙 진입 → 종료 1주.

## 0. 트랙 목표
OWASP Top 10 (2021) 기준 v1 적용 가능한 항목 baseline 박기. v2 Supabase Auth 진입 전 안전망.

## 1. CP 별 결과 요약

| CP | 카테고리 | 결과 | 산출물 |
|---|---|---|---|
| CP238 | A06 Dependency CVE | pip-audit 0 / npm audit 0 (high+) | docs/cp238_*_report.md, ADR-0029 |
| CP239 | A02 Secrets | gitleaks history 0 (clean) | docs/cp239_*_report.md, rotation runbook, ADR-0030 |
| CP240 | A05 Security Headers | securityheaders.com A | docs/cp240_*_report.md, ADR-0031 |
| CP241 | A03 Injection (input validation) | TickerStr + negative test 6+ | docs/cp241_*_report.md, ADR-0032 |
| CP242 | A04/A05 CORS + rate limit | CorsConfig 점검 + slowapi 60/min | docs/cp242_*_report.md, ADR-0033 |

## 2. OWASP Top 10 (2021) 대응 현황

| # | 카테고리 | 대응 상태 | 위치 |
|---|---|---|---|
| A01 Broken Access Control | ⚪ v1 인증 없음 (v2 RLS) | — |
| A02 Cryptographic Failures | ✅ CP239 | secrets rotation runbook |
| A03 Injection | ✅ CP241 + SQLAlchemy ORM (CP236b) | TickerStr / SearchStr |
| A04 Insecure Design | ✅ CP242 | rate limit baseline |
| A05 Security Misconfiguration | ✅ CP240 + CP242 | 보안 헤더 + CORS |
| A06 Vulnerable Components | ✅ CP238 | pip-audit + npm audit + Dependabot |
| A07 Auth Failures | ⚪ v1 인증 없음 (v2 Supabase Auth) | — |
| A08 Integrity Failures | ✅ CP237 CI + lockfile | requirements==, package-lock.json |
| A09 Logging Failures | ✅ CP228 + CP239 | structlog + gitleaks |
| A10 SSRF | ✅ 해당 없음 | user URL input 없음 |

→ v1 대응 가능 항목 **100%** 처리. 인증 의존 항목 (A01/A07) 은 v2 트랙.

## 3. 외부 검증 결과

- securityheaders.com (frontend): A
- securityheaders.com (backend): A
- Mozilla Observatory: <점수>
- pip-audit: 0 high+
- npm audit: 0 high+
- gitleaks: 0 finding (clean)

## 4. 산출물

- 5 ADR (0029~0033)
- 5 CP 보고서 (cp238~cp242)
- 신규 파일:
  - backend/app/core/security_headers.py
  - backend/app/core/validators.py
  - backend/app/core/rate_limit.py
  - backend/tests/test_cp241_input_validation.py
  - backend/tests/test_cp242_rate_limit.py
  - frontend/next.config.js (headers)
  - docs/security/rotation_runbook.md
- 수정:
  - backend/app/main.py (middleware add)
  - backend/app/config/settings.py (CorsConfig regex)
  - backend/requirements.txt (slowapi)
  - backend/requirements-dev.txt (pip-audit)
  - .github/workflows/ci.yml (audit + gitleaks job)
  - .pre-commit-config.yaml (gitleaks hook)

## 5. CI 게이트 최종 상태

| Job | 상태 |
|---|---|
| backend ruff | GREEN |
| backend mypy | continue-on-error (baseline) |
| backend pytest | GREEN |
| backend CP223 snapshot | GREEN |
| **backend pip-audit (CP238)** | GREEN |
| frontend tsc | GREEN |
| frontend vitest | GREEN |
| frontend playwright | GREEN |
| **frontend npm audit (CP238)** | GREEN |
| **secrets gitleaks (CP239)** | GREEN |

## 6. v2 보안 트랙 예고

- A01 RLS 본격 (Supabase Row Level Security)
- A07 Auth Failures (JWT, refresh, MFA)
- Rate limit 본격 (login brute force, path 별 차등)
- CSP nonce 적용 (`'unsafe-inline'` 제거 → A+ 도전)
- Sentry 연동 (보안 이벤트 알람)
- HSTS preload 등록 결정
- Subresource Integrity (SRI) for CDN

## 7. 회고

- 가장 가치 컸던 CP: CP240 (외부 검증 가능한 A 등급)
- 가장 시간 든 CP: <후기>
- 발견된 P0/P1: <기록>
```

### Step 9 — ADR-0033

`docs/adr/0033_cors_rate_limit_security_track_close.md`:

```markdown
# ADR-0033: CORS Policy + Rate Limit + Security Track Close

## Status
Accepted (2026-06-06)

## Context
CP238~241 완료 후 마지막 CP: CORS 안전 점검 + rate limit 도입 결정 + 보안 트랙 종합 회고.

## Decision

### 1. CORS
- `BACKEND_CORS_ORIGINS` env: 명시 origin 만. production 에서 `*` fallback 금지 (settings 검증 추가).
- `BACKEND_CORS_ORIGIN_REGEX`: Vercel preview 도메인 패턴 `^https://lens(-[a-z0-9-]+)?\.vercel\.app$`.
- `allow_credentials=True` 유지 (cookie/session 향후 사용 대비). `*` 와 동시 사용 금지.

### 2. Rate Limit (v1 가볍게 도입)
- `slowapi` (in-memory).
- prediction endpoint: 60/minute.
- search: 30/minute.
- `/health/live`: 무제한 (UptimeRobot 호환).
- 429 응답 schema: 통일 (CP241 패턴).
- 다중 인스턴스 진입 시 Redis 로 storage 옮김 (v2).

### 3. 보안 트랙 회고
- OWASP Top 10 v1 적용 가능 항목 100% 대응.
- v2 트랙: A01 RLS + A07 Auth + Sentry + CSP nonce + SRI.

## Consequences
- CORS regex 로 preview 도메인 자동 허용 → 새 preview 마다 env 수정 불필요
- rate limit 으로 Render free tier 폭주 방어 baseline
- v2 진입 시 `slowapi` 를 redis backend 로 교체 필요 (다중 인스턴스 대비)
- securityheaders.com A 등급 유지 / CI 게이트 5종 GREEN

## References
- CP238~241 ADR (0029~0032)
- OWASP Top 10
- slowapi https://github.com/laurentS/slowapi
```

---

## 5. 회귀 안전망

- **CP223 BE snapshot**: CORS / rate limit 추가는 응답 schema 변경 없음 (header 만) → 0 diff
- **CP230 FE smoke**: 정상 흐름 회귀 0
- Step 6, 7 의 rate limit pytest 가 새 안전망

---

## 6. 성공 기준 (L8)

- `BACKEND_CORS_ORIGINS` production 값에 `*` 0 occurrence
- Vercel preview 도메인 (`lens-XXX.vercel.app`) curl OPTIONS 검증 200
- (rate limit 도입 시) 61회 호출 시 429 발생 + `/health/live` 100회 호출 200
- CI workflow 모든 job GREEN
- `docs/cp242_security_track_summary.md` 작성
- ADR-0033 작성

---

## 7. 인터페이스 보존 (L7)

- 정상 origin 요청 영향 0
- 정상 횟수 요청 영향 0 (60/min 안)
- 429 응답 schema 는 통일된 error schema 와 정합 → frontend 가 처리 가능

---

## 8. Lens 특화 (L9)

- **UptimeRobot 호환** — `/health/live` 무제한 보장
- **Render free tier sleep cold start** — rate limit 이 첫 ping 막으면 안 됨 → `/health/*` 예외
- **Vercel preview 도메인 변동** — regex 로 처리, env 수동 갱신 불필요
- 운영 모델 3개 추론 endpoint 가 rate limit 영향 받음 → 사용자 (본인) 가 평가 시 분당 60회 안 넘는지 확인 (충분)

---

## 9. ADR-0033 작성 가이드

§4 Step 9 참조.

---

## 10. 자동 실행 적합도

| Step | 자동 | 사람 확인 |
|---|---|---|
| 1 | ✅ | — |
| 2 | ❌ | **Render dashboard 수동 확인** |
| 3 | ✅ | — |
| 4 | ✅ | — |
| 5 | ❌ | **rate limit 도입 결정 사용자** |
| 6 | ✅ | (도입 결정 후) |
| 7 | ✅ | — |
| 8 | ✅ | — |
| 9 | ✅ | — |

---

## 11. 종료 후 commit / 보고

### 권장 commit 분할

```
CP242 Step 1: CorsConfig audit (no code change | add production validator)
CP242 Step 3: main.py CORSMiddleware audit (no change | fix credentials+origins combo)
CP242 Step 4: BACKEND_CORS_ORIGIN_REGEX for Vercel previews
# 도입 결정 시:
CP242 Step 6: add slowapi + core/rate_limit.py
CP242 Step 7: apply @limiter.limit + /health/* exception + 429 handler
CP242 Step 7 test: rate limit + health unlimited pytest
CP242 security track summary (cp238~242)
CP242 ADR-0033 (CORS + rate limit + track close)
```

### 보고서
- `docs/cp242_cors_rate_limit_report.md` (CP242 단일)
- `docs/cp242_security_track_summary.md` (CP238~242 종합)

### ADR
`docs/adr/0033_cors_rate_limit_security_track_close.md`

---

**진입 조건**: CP238~CP241 완료. CP235 Pydantic Settings (CorsConfig) 존재.
**다음 CP**: 없음 (보안 트랙 종료).
**리스크**: Step 2 Render dashboard 확인 누락 시 production CORS 가 `*` 인 채로 남을 수 있음 → agent 가 사용자에게 명시적 alarm.
