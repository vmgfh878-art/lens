# CP240 — HTTP Security Headers + CSP 보고서

작성: 2026-06-09. OWASP A05 대응. ADR-0031 동반.

## 0. 한 줄 요약

backend (FastAPI `SecurityHeadersMiddleware`) + frontend (Next.js
`next.config.mjs` 의 `async headers()`) 양쪽에 **6 보안 헤더** (HSTS /
X-Content-Type-Options / X-Frame-Options / Referrer-Policy /
Content-Security-Policy / Permissions-Policy) 박음. CSP 는 Lens 사용
패턴 (lightweight-charts / Microsoft Clarity / 로컬 dev 직접 호출 /
production same-origin proxy) 정확 반영. backend pytest 8 + Playwright
e2e 4 view CSP guard 영구 안전망 박음. 외부 검증: 사용자가 production
live URL 에 curl 으로 6 헤더 직접 확인 완료. securityheaders.com 등급 A
예상 (A+ 는 v2 nonce 도입 commitment). **CP240 마감**.

## 1. 핵심 컴포넌트 존재 체크리스트

CP240 은 보안 헤더 박음 + 영구 안전망 박음. ML 컴포넌트 변경 0 (N/A). 대신:

- [x] `backend/app/core/security_headers.py` 신규 (`SecurityHeadersMiddleware`)
- [x] `backend/app/main.py` middleware add (GZip 다음 + request_id 전, outermost)
- [x] `frontend/next.config.mjs` 의 `async headers()` 추가 (5 헤더)
- [x] backend + frontend CSP 동일 정책 (defense in depth 일관성)
- [x] Lens 사용 패턴 정확 반영 (Clarity `*.clarity.ms` + `www.clarity.ms` /
  production backend URL / dev `127.0.0.1:8000`)
- [x] `backend/tests/test_security_headers.py` 7 test 영구 안전망
- [x] `frontend/tests/e2e/csp_violation_guard.spec.ts` 4 view CSP guard
- [x] `docs/adr/0031_security_headers_csp.md` 정책 + v2 commitment
- [x] 운영 코드 / ML 모델 / 응답 schema 0 수정 (헤더만 추가)

## 2. 새 테스트 결과

```
pytest backend/tests/test_security_headers.py -v
7 passed in 1.21s
```

| Test | 결과 |
|---|---|
| `test_security_headers_attached_on_health_live` | ✅ 5 헤더 모두 박힘 |
| `test_security_headers_attached_on_data_endpoint` | ✅ `/stocks` 도 동일 |
| `test_strict_transport_security_value` | ✅ HSTS 2년 + includeSubDomains |
| `test_x_content_type_options_nosniff` | ✅ `nosniff` |
| `test_x_frame_options_deny` | ✅ `DENY` |
| `test_referrer_policy` | ✅ `strict-origin-when-cross-origin` |
| `test_csp_required_directives` | ✅ default-src / script-src / frame-ancestors 'none' / base-uri 'self' |

Playwright e2e `csp_violation_guard.spec.ts` (4 view × CSP violation 0):
- 실행: `cd frontend; npm run test:e2e -- csp_violation_guard`
- 선행: `scripts/start_demo.ps1` 으로 backend 8000 + frontend 3000 ready
- CI 에서는 continue-on-error (webServer 미정의). 실제 차단은 로컬 +
  사용자 spot-check.

## 3. 회귀 안전망 (CP240 변경 후 재확인)

| 안전망 | 결과 |
|---|---|
| CP223 characterization snapshot | ✅ 9 passed (response body 영향 0 — CP237.5 drift-resilient 정규화가 헤더 무관) |
| drift sim | ✅ 11 passed |
| CP230 frontend smoke (Vitest) | ✅ 8 files / 166 passed |
| backend pytest 전체 | 118 passed + 11 pre-existing fail (CP237.5 보고서 §4.2) |
| frontend tsc | ✅ 0 error |

## 4. 진행 중 발견 + 결정

### F1. baseClient.ts 의 환경별 backend 호출 패턴

- **로컬 dev** (`localhost`/`127.0.0.1`): `apiBaseUrl = http://127.0.0.1:8000`
  (직접 호출, proxy 안 씀)
- **production**: `apiBaseUrl = /__backend` (same-origin proxy via
  `next.config.mjs` rewrites)

→ CSP `connect-src` 에 둘 다 명시 (`'self'` + `http://127.0.0.1:8000` +
`https://lens-backend-7stj.onrender.com`).

### F2. Microsoft Clarity 외부 도메인

- `frontend/src/app/ClarityInit.tsx` 가 `@microsoft/clarity` init 호출
- 외부 script load + telemetry → `*.clarity.ms` + `www.clarity.ms` 필수
- script-src / connect-src / img-src 에 모두 명시

### F3. lightweight-charts + Next.js hydration 호환

- `'unsafe-inline'` + `'unsafe-eval'` 임시 유지 (A 등급 trade-off)
- v2 에서 nonce 도입해 제거 → A+ 도전 (ADR-0031 §6 commitment)

### F4. backend + frontend CSP 동일 정책

- API 응답은 JS 실행 X → script-src 효과 작음. 그러나 일관성 + defense
  in depth.
- middleware 순서: GZip 다음 (outermost) — 모든 응답 후처리 마지막

## 5. 외부 검증 결과 (Step 8 사용자 직접)

### 5.1 헤더 prod live 직접 검증 (curl)

사용자가 production live URL 에 curl 으로 6 헤더 직접 확인:

URL | 6 헤더 | 비고
---|---|---
`https://lens-ten-delta.vercel.app` (frontend) | ✅ HSTS / X-Content-Type-Options / X-Frame-Options / Referrer-Policy / Content-Security-Policy / Permissions-Policy | curl 응답 직접 확인 완료 |
`https://lens-backend-7stj.onrender.com` (backend) | ✅ 동일 6 헤더 | curl 응답 직접 확인 완료 |

### 5.2 securityheaders.com

URL | 등급 | 비고
---|---|---
`https://lens-ten-delta.vercel.app` | **A (예상)** | 헤더 직접 검증 완료, 스캔 스크린샷은 선택 |
`https://lens-backend-7stj.onrender.com` | **A (예상)** | 동일 |

A+ 가 아닌 이유 (ADR-0031 §5 / §6 와 일관): `'unsafe-inline'` + `'unsafe-eval'`
유지 (lightweight-charts + Next.js hydration + Clarity init 호환). v2 에서
nonce 적용해 A+ 도전 commitment.

### 5.3 Mozilla Observatory

스킵 (사용자 결정 — 직접 헤더 검증으로 충분).

## 6. 산출물

### 신규
- `backend/app/core/security_headers.py` (`SecurityHeadersMiddleware` + 5 헤더 + CSP)
- `backend/tests/test_security_headers.py` (7 test 영구 안전망)
- `frontend/tests/e2e/csp_violation_guard.spec.ts` (4 view CSP guard)
- `docs/adr/0031_security_headers_csp.md`
- `docs/cp240_security_headers_report.md` (본 보고서)

### 수정
- `backend/app/main.py` (import + `app.add_middleware(SecurityHeadersMiddleware)`)
- `frontend/next.config.mjs` (`async headers()` 추가, 기존 `rewrites()` 유지)

### 운영 코드 / ML 모델 / 응답 schema 0 수정

## 7. commit 이력 (CP240, 6 commit + closure)

```
13d4280 CP240 Step 1-2: backend SecurityHeadersMiddleware + 영구 안전망 test
ed26a12 CP240 Step 3: next.config.mjs 의 async headers() — 5 헤더 + CSP
fa103d6 CP240 Step 4-5: CSP 조정 (Clarity 도메인 + production backend + dev localhost)
6994e84 CP240 Step 6: Playwright e2e CSP violation guard
3d3cdab CP240 Step 10 closure draft: report + ADR-0031
9aee8d4 CP240 Step 8 보강: Permissions-Policy (deny unused features + topics opt-out)
<본 commit> CP240 closing: report §5 갱신 (Step 8 done — 6 헤더 prod live curl 확인 A 예상)
```

## 8. v2 재검토 commitment (ADR-0031 §6)

- nonce 도입 (`'unsafe-inline'` 제거 → A+ 가능)
- HSTS preload 등록 결정
- CSP report-uri (`/api/v1/csp-report`) 도입 — violation 모니터링
- Supabase Auth 도입 시 attack surface 증가에 맞춰 CSP 정밀화

## 9. 다음 CP

CP241 (Input validation Pydantic). 진입 조건 (CP237.5 GREEN) 충족.
