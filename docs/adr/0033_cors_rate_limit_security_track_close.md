# ADR-0033: CORS Policy + Rate Limit + Security Track Close

Status: Accepted
Date: 2026-06-09
CP: CP242 (OWASP A04/A05 + 보안 트랙 종료)

## Context

CP238~241 완료 후 마지막 CP: CORS 안전 점검 + rate limit 도입 결정 + 보안 트랙
종합 회고. v2 (Supabase Auth + production 본격화) 진입 전 baseline 마무리.

## Decision

### 1. CORS

- `BACKEND_CORS_ORIGINS` env: 명시 origin 만 허용.
  `_DEFAULT_CORS_ORIGINS` = `http://localhost:3000,http://127.0.0.1:3000,
  https://lens-kimjihyeong-s-projects.vercel.app,https://lens-ten-delta.vercel.app`
  (CP235 시점 박힘).
- **CP242 보강**: `CorsConfig.raw_origins` 의 `field_validator
  _forbid_wildcard` 추가 — env 에 `*` 박힌 경우 fail-loud (ValueError).
  production / dev 무관 single guard.
- `BACKEND_CORS_ORIGIN_REGEX` = `^https://lens(?:-[a-z0-9-]+)?\.vercel\.app$` —
  Vercel preview 도메인 자동 허용 (CP235 시점 박힘).
- **`main.py` CORSMiddleware**: `allow_credentials=False` — `*` 와 조합해도
  spec 위반 X (현재 명시 origin 만). cookie/session 도입 시 (v2) `True` 로
  바꾸면 `*` 와 동시 사용 금지 강제.

### 2. Rate Limit — v1 보류

사용자 결정 (CP242 Step 5 R3): **v1 보류, v2 Auth 도입 시 함께 진행**.

근거:
- v1 사용자 본인 + 평가자만 사용 → DoS 위험 작음
- slowapi 의존성 추가 대비 보안 baseline 충분 (CP240 보안 헤더 + CP241 input
  validation + CP242 CORS strict)
- v2 Supabase Auth 도입 시 login brute force 방어가 주된 rate limit 시나리오 →
  그 때 path 별 차등 + Redis backend 함께 검토

v2 계획:
- `slowapi` (in-memory) → 단일 인스턴스 baseline
- 다중 인스턴스 진입 시 Redis storage 로 옮김
- prediction endpoint: 60/minute
- search: 30/minute
- `/health/live`: 무제한 (UptimeRobot 호환)
- Auth login endpoint: 5/minute (brute force 방어)
- 429 응답 schema: 통일 (CP241 패턴)

### 3. 보안 트랙 회고

OWASP Top 10 v1 적용 가능 항목 **100% 대응** (인증 의존 A01/A07 제외).
6 CP (CP237.5 prereq + CP238~CP242) / 5 ADR / 6 보고서 + 1 종합 보고서.
51 backend pytest 영구 안전망 + frontend Vitest 166 + Playwright CSP guard.

v2 트랙 (Supabase Auth + production 본격화) 진입 조건 충족 → **트랙 종료**.

## Consequences

### 장점
- CORS `*` env 박을 위험 fail-loud 차단 (운영 실수 차단)
- Vercel preview 도메인 자동 허용 (env 수동 갱신 불필요)
- securityheaders.com A 등급 유지 / CI 게이트 5종 GREEN
- 51 backend pytest 영구 안전망 → 향후 분리 리팩토링 회귀 검출
- 보안 트랙 종합 보고서 (`docs/cp242_security_track_summary.md`) 로 OWASP
  Top 10 v1 대응 현황 한 곳에서 추적 가능

### 단점 / Trade-off
- rate limit 보류 → v1 운영 중 DoS 시나리오 발생 시 즉시 대응 못 함 (단,
  사용자 본인 + 평가자만 사용 가정)
- CP239 gitleaks 미작동 acknowledged (defense in depth 약화) — `.gitignore`
  + 수동 spot-check + 정기 rotation 으로 대체
- CP238 frontend 7 acknowledged advisories — v2 next 16 major bump 시 정리
- CP240 CSP `'unsafe-inline'` + `'unsafe-eval'` — v2 nonce 도입 시 A+ 도전

## v2 트랙 예고

- A01 RLS (Supabase Row Level Security)
- A07 Auth (JWT, refresh, MFA)
- Rate limit 본격 도입 (slowapi → Redis)
- CSP nonce 적용 (A+ 도전)
- gitleaks 재시도 (trufflehog / shallow grep script)
- next 16 major bump (RSC 7 acknowledged 해결)
- HSTS preload 등록 결정
- Sentry 연동
- Subresource Integrity (SRI)

## References

- CP237.5 ~ CP241 ADR (0028.5, 0029~0032)
- OWASP Top 10 (2021)
- slowapi (https://github.com/laurentS/slowapi) — v2 대상
- 본 트랙 종합 보고서: `docs/cp242_security_track_summary.md`
- 본 CP 단일 보고서: `docs/cp242_cors_rate_limit_report.md`
