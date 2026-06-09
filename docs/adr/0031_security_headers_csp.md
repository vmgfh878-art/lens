# ADR-0031: HTTP Security Headers + CSP Policy

Status: Accepted
Date: 2026-06-09
CP: CP240 (OWASP A05)

## Context

Lens v1 운영 진입 직전 응답 보안 헤더 0. securityheaders.com 추정 등급 F.
OWASP A05 Security Misconfiguration 무방비. v2 (Supabase Auth + production
본격화) 진입 전 baseline 박아야.

## Decision

### 1. 5 헤더 박음 (backend FastAPI + frontend Next.js 양쪽)

| 헤더 | 값 |
|---|---|
| `Strict-Transport-Security` | `max-age=63072000; includeSubDomains` (2년) |
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Content-Security-Policy` | (§3 참조) |

### 2. HSTS preload list 등록은 보류

`max-age=63072000` 만으로 충분. preload list 등록은 cancel 어려움 (수개월
대기) → v2 결정.

### 3. CSP 정책

```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval'
  https://www.clarity.ms https://*.clarity.ms;
style-src 'self' 'unsafe-inline';
img-src 'self' data: https://*.clarity.ms;
font-src 'self' data:;
connect-src 'self' https://www.clarity.ms https://*.clarity.ms
  https://lens-backend-7stj.onrender.com http://127.0.0.1:8000;
frame-ancestors 'none';
base-uri 'self';
form-action 'self'
```

#### 정책 근거

- **`'unsafe-inline'`**: lightweight-charts + Next.js hydration + Microsoft
  Clarity init 호환 (임시). 제거 시 hydration 깨짐 → Playwright e2e
  `csp_violation_guard.spec.ts` 가 회귀 감지. v2 에서 nonce 적용해 제거 검토.
- **`'unsafe-eval'`**: lightweight-charts 의 일부 동적 코드 평가 호환. v2
  검토.
- **`*.clarity.ms` + `www.clarity.ms`**: Microsoft Clarity (외부 telemetry).
  둘 다 명시 — 브라우저별 wildcard 처리 차이 대비 (안전 우선).
- **`lens-backend-7stj.onrender.com`**: production backend URL (직접 호출
  대비). 보통은 same-origin proxy (`/__backend/*` via `next.config.mjs`
  rewrites) 라 `'self'` 가 처리.
- **`127.0.0.1:8000`**: 로컬 dev 의 `baseClient.ts` localhost 분기 (frontend
  3000 → backend 8000 직접 호출).
- **`frame-ancestors 'none'`**: X-Frame-Options DENY 보강 (구형 브라우저
  호환).
- **`base-uri 'self'`**: base tag 우회 방지.
- **`form-action 'self'`**: form submit 대상 제한.

### 4. Backend + Frontend 동일 정책 (defense in depth)

API 응답은 JSON 만 반환이라 JS 실행 X — script-src / connect-src 효과
작음. 주된 효과는 frontend (`next.config.mjs`) 의 `headers()`. 단 backend
도 일관성 + defense in depth 차원에서 동일 정책. middleware 순서: GZip
다음 + request_id_middleware 전 (outermost, 응답 후처리 마지막).

### 5. 회귀 안전망

- **`backend/tests/test_security_headers.py`** (영구 안전망, 7 test):
  5 헤더 존재 + 5 헤더별 정확한 값 + CSP 필수 directive 4개.
  `SecurityHeadersMiddleware` 가 끊기면 즉시 RED.
- **`frontend/tests/e2e/csp_violation_guard.spec.ts`** (Playwright):
  4 view 별 CSP violation console error 0 검증. lightweight-charts /
  Clarity / Next.js hydration 호환 검증.
- **외부 검증** (Step 8/9 사용자 직접):
  - https://securityheaders.com — A 등급 목표 (A+ 는 v2 nonce 필요)
  - https://observatory.mozilla.org — 75+ 목표

### 6. v2 재검토 commitment

- nonce 도입 (`'unsafe-inline'` 제거 → A+ 가능)
- HSTS preload 등록 결정
- CSP report-uri (`/api/v1/csp-report`) 도입 — violation 모니터링
- Supabase Auth 도입 시 attack surface 증가에 맞춰 CSP 정밀화

## Consequences

### 장점
- securityheaders.com A 등급 (목표) — OWASP A05 baseline 박힘
- 5 헤더 일관 backend + frontend (defense in depth)
- 회귀 안전망 backend pytest (7) + Playwright e2e (4 view) 박힘
- Clarity / lightweight-charts / backend 호출 모두 명시 → CSP 누락 0

### 단점 / Trade-off
- `'unsafe-inline'` + `'unsafe-eval'` 허용 → A+ 못 감 (A 머무름). XSS 방어
  약화 — v1 사용자 입력 거의 없음 (ticker 만) 으로 위험 완화. v2 nonce
  도입 commitment.
- Playwright e2e 는 CI 에서 `continue-on-error` (webServer 미정의) → 실제
  차단은 로컬 개발자 + 사용자 spot-check 가 주.
- HSTS 2년 박음 → 잘못 박으면 2년 cancel 어려움. preload 등록 안 함이
  최소한의 fallback.

## References

- OWASP Top 10 A05
- MDN Web Security Headers (https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers#security)
- web.dev/csp
- securityheaders.com grading criteria
- 본 트랙 보고서: `docs/cp240_security_headers_report.md`
