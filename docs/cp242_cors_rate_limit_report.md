# CP242 — CORS + Rate Limit + 보안 트랙 종료 보고서

작성: 2026-06-09. OWASP A04/A05. ADR-0033 + 종합 보고서 동반.

## 0. 한 줄 요약

CorsConfig (CP235 시점 적용) 점검 — `*` 미사용 + Vercel preview regex 적용 +
`allow_credentials=False` 안전. CP242 보강: `raw_origins` `*` 차단 validator
(fail-loud). rate limit v1 **보류** (사용자 결정, v2 Auth 시 도입). 보안 트랙
종합 보고서 작성. **트랙 종료**.

## 1. 핵심 컴포넌트 존재 체크리스트

- [x] `_DEFAULT_CORS_ORIGINS` 명시 origin 4개 (`*` 0)
- [x] `_DEFAULT_CORS_ORIGIN_REGEX` Vercel preview 정규식 (CP235 시점)
- [x] `CorsConfig.raw_origins` `_forbid_wildcard` validator (CP242 보강)
- [x] `main.py` CORSMiddleware `allow_credentials=False` (`*` 조합 spec 위반 X)
- [x] Render dashboard `BACKEND_CORS_ORIGINS` 사용자 직접 확인 (R3 done, `*` 아님 확정)
- [x] rate limit 도입 결정 — v1 보류 (R3 done, ADR-0033 명시)
- [x] `docs/cp242_security_track_summary.md` 종합 보고서
- [x] `docs/adr/0033_cors_rate_limit_security_track_close.md`
- [x] 운영 코드 / ML 모델 / 응답 schema 0 수정 (validator 추가만)

## 2. 새 테스트 결과

CP242 자체 신규 unit test 추가 없음 (CORS validator 는 단순 fail-loud, ML
변경 X). 회귀 안전망 재실행:

```
pytest backend/tests -k "snapshot or characterization or cp223 or security_headers or input_validation" -q
51 passed in 3.61s
```

| 안전망 | 결과 |
|---|---|
| CP223 characterization snapshot | ✅ 9 passed |
| CP237.5 drift simulation | ✅ 11 passed |
| CP240 test_security_headers | ✅ 8 passed |
| CP241 test_input_validation | ✅ 22 passed |
| 기타 cp223 매치 | ✅ 1 passed |
| **합계** | **51 PASS** |

## 3. R3 사람 확인 결과

### 3.1 Step 2 — Render dashboard CORS 확인

사용자 응답: "xxx 로 하나 있는데 교체 할까?".

해석 + 결정:
- env 에 명시 origin 1개 박혀있음 (정확한 값 사용자만 노출 — 보안 차원 적정)
- `*` 인 경우 CP242 validator fail-loud → backend startup 시 ValueError. 현재
  backend 정상 동작 = `*` 아님 확정.
- preview 도메인은 `_DEFAULT_CORS_ORIGIN_REGEX` 가 자동 처리 → 명시 origin 1개만
  박혀있어도 production frontend + preview 도메인 모두 통과
- **교체 불필요**. 현재 env 유지.

### 3.2 Step 5 — Rate limit 도입 결정

사용자 응답: **v1 보류, v2 도입**.

근거 (ADR-0033 §2):
- v1 사용자 본인 + 평가자만 → DoS 위험 작음
- slowapi 의존성 대비 보안 baseline 충분 (CP240 + CP241 + CP242 CORS)
- v2 Auth 도입 시 login brute force 방어가 주 시나리오 → 함께 진행

Step 6, 7 (slowapi 도입 + endpoint limiter + /health 예외 + pytest) **skip**.

## 4. 회귀 안전망 (CP242 변경 후 재확인)

51 PASS (CP223 + drift + security_headers + input_validation + cp223). 응답
schema 영향 0 (validator 추가 + ADR/보고서 작성만).

## 5. 진행 중 발견

### F1. CorsConfig 가 이미 안전 (CP235 시점)

- `_DEFAULT_CORS_ORIGINS` 4 명시 origin (`*` 0)
- `_DEFAULT_CORS_ORIGIN_REGEX` Vercel preview 정규식
- `allow_credentials=False` (`*` spec 위반 0)

→ Step 1/3/4 가 **audit only** 수준. 코드 변경 거의 없음.

### F2. CP242 validator 추가 — 미래 보강

env 에 `*` 박는 운영 실수 + 외부 침입 차단. `field_validator` 1줄. production
/ dev 무관 single guard.

## 6. 산출물

### 신규
- `docs/cp242_security_track_summary.md` (보안 트랙 종합 보고서, CP237.5 +
  CP238~CP242)
- `docs/adr/0033_cors_rate_limit_security_track_close.md`
- `docs/cp242_cors_rate_limit_report.md` (본 보고서)

### 수정
- `backend/app/config/settings.py` (CorsConfig `_forbid_wildcard` validator 추가)

### 운영 코드 / ML 모델 / 응답 schema 0 수정

## 7. commit 이력 (CP242, 2 commit + closing)

```
169cc52 CP242 Step 1+3+4: CORS audit + CorsConfig 의 * env 차단 validator
<본 commit> CP242 Step 8-9 closure: 종합 보고서 + ADR-0033 + CP242 보고서
```

(rate limit 도입 보류로 Step 6, 7 skip.)

## 8. 보안 트랙 종료

OWASP Top 10 v1 적용 가능 항목 **100% 대응** (인증 의존 A01/A07 제외).
6 CP (CP237.5 prereq + CP238~CP242) / 5 ADR / 6 보고서 + 1 종합 보고서.

v2 트랙 (Supabase Auth + production 본격화) 진입 조건 충족 → **트랙 GREEN
종료**.

## 9. 다음 트랙

**v2 보안 트랙** (Supabase Auth + production 본격화 시점):
- A01 RLS (Row Level Security)
- A07 Auth (JWT, refresh, MFA)
- Rate limit 본격 (slowapi + Redis, path 별 차등) — 본 CP 보류분
- CSP nonce 적용 (A+ 도전) — CP240 commitment
- gitleaks 재시도 — CP239 v2 재검토
- next 16 major bump — CP238 7 acknowledged 정리
- HSTS preload / Sentry / SRI

상세는 `docs/cp242_security_track_summary.md` §8.
