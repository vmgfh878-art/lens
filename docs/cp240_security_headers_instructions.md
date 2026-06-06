# CP240 지시서 — HTTP 보안 헤더 5종 + securityheaders.com A 등급 (OWASP A05)

> 작성: 2026-06-06. 트랙: Lens 보안 트랙. ADR-0031 동반.
> 사용자 환경: Windows PowerShell. 한국어 보고서.

---

## 0. 한 줄 목표

`Strict-Transport-Security` / `Content-Security-Policy` / `X-Frame-Options` / `X-Content-Type-Options` / `Referrer-Policy` 5개 헤더를 backend (FastAPI) + frontend (Next.js) 양쪽 응답에 박아 [securityheaders.com](https://securityheaders.com) **A 등급** 달성. lightweight-charts 시각 회귀 0.

---

## 1. 진단

| 헤더 | 현재 | 효과 |
|---|---|---|
| `Strict-Transport-Security` (HSTS) | 없음 | 첫 방문 HTTP → HTTPS 강제 (downgrade 차단) |
| `Content-Security-Policy` (CSP) | 없음 | XSS 발생해도 외부 스크립트 차단 |
| `X-Frame-Options` | 없음 | iframe 삽입 금지 (clickjacking) |
| `X-Content-Type-Options` | 없음 | MIME sniffing 차단 |
| `Referrer-Policy` | 없음 | 외부 이동 시 URL 노출 제어 |
| securityheaders.com 추정 등급 | F (헤더 0) | — |

**OWASP**: A05 Security Misconfiguration.

---

## 2. 변경 내용

- **Backend**: `backend/app/core/security_headers.py` 신규 → `SecurityHeadersMiddleware` 정의 → `main.py` 에 add
- **Frontend**: `frontend/next.config.js` (또는 `.mjs`) 에 `async headers()` 추가
- **CSP 정책**: `script-src` 신중 (`'unsafe-inline'` 임시 허용, lightweight-charts 동작 보장), `connect-src` 에 Render backend URL 명시
- Playwright smoke 통과 (CSP 로 차트 안 깨지는지)
- [securityheaders.com](https://securityheaders.com) A 등급 + Mozilla Observatory second opinion
- ADR-0031 작성

---

## 3. Step 분할

| Step | 내용 | 위험 | 시간 | 자동/수동 |
|---|---|---|---|---|
| 1 | `backend/app/core/security_headers.py` 신규 (`SecurityHeadersMiddleware`) | 매우낮음 | 30분 | 자동 |
| 2 | `main.py` 에 middleware add → 로컬 backend 띄워 curl 로 header 확인 | 낮음 | 15분 | 자동 |
| 3 | `frontend/next.config.js` 에 `async headers()` 추가 (frontend 응답에도) | 매우낮음 | 15분 | 자동 |
| 4 | 로컬 `npm run dev` → DevTools Network 탭에서 header 확인 | 낮음 | 10분 | 반자동 (사용자 확인) |
| 5 | **CSP 조정**: lightweight-charts 가 inline script 쓰는지 확인 → `script-src 'self' 'unsafe-inline'` 으로 시작 → 더 좁힐 수 있는지 검토 (nonce/hash) | 🔴 큼 (UI 깨질 가능) | 1h | 반자동 |
| 6 | Playwright smoke 통과 확인 (`npm run test:e2e`) — 차트 element 정상 렌더링 | 낮음 | 15분 | 자동 |
| 7 | Render redeploy (backend) → Vercel redeploy (frontend) | 낮음 | 자동 | — |
| 8 | https://securityheaders.com 에 production URL 검사 → A 등급 확인 | 매우낮음 | 5분 | **수동** (사용자가 URL 직접 입력) |
| 9 | https://observatory.mozilla.org second opinion | 매우낮음 | 5분 | **수동** |
| 10 | `docs/cp240_security_headers_report.md` + ADR-0031 | 매우낮음 | 30분 | 자동 |

---

## 4. 각 Step 정확한 명령 / 코드

### Step 1 — Backend middleware

`backend/app/core/security_headers.py` (신규):

```python
"""CP240 — HTTP 보안 헤더 미들웨어.

5 헤더 박음:
- Strict-Transport-Security: HTTPS 강제 (2년 + subdomain 포함)
- X-Content-Type-Options: MIME sniffing 차단
- X-Frame-Options: clickjacking 차단 (iframe 금지)
- Referrer-Policy: 외부 이동 시 URL 누출 제어
- Content-Security-Policy: XSS / 외부 스크립트 차단

CSP 는 API 응답에도 박지만 사실 API 는 JS 안 실행이라 효과 작음.
주된 효과는 frontend (Vercel) 의 next.config.js headers() 에서 나온다.
다만 일관성 + defense in depth 차원에서 API 응답에도 박는다.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# CSP — API 응답용. frontend 와 정합 유지.
# script-src 'unsafe-inline' 은 lightweight-charts inline script 호환 (임시).
#   향후 nonce/hash 도입 시 'unsafe-inline' 제거 가능 → ADR-0031 명시.
# connect-src 에 Render backend URL 박음 (frontend 가 cross-origin 호출).
_API_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self' https://lens-backend-7stj.onrender.com; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

_HEADERS: dict[str, str] = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": _API_CSP,
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """모든 응답에 보안 헤더 5종 부착."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for key, value in _HEADERS.items():
            # 기존 헤더 있으면 덮어쓰지 않음 (별도 미들웨어가 더 strict 한 정책 박을 수 있음)
            response.headers.setdefault(key, value)
        return response
```

### Step 2 — main.py 에 add

`backend/app/main.py` 의 `CORSMiddleware` add 다음 줄에:

```python
from app.core.security_headers import SecurityHeadersMiddleware

# ... 기존 코드 ...
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(SecurityHeadersMiddleware)   # CP240
```

검증:
```powershell
cd C:\Users\user\lens\backend
uvicorn app.main:app --reload --port 8000

# 다른 창에서:
curl -I http://127.0.0.1:8000/api/v1/health/live
```

기대 출력 (헤더 5개 포함):
```
HTTP/1.1 200 OK
strict-transport-security: max-age=63072000; includeSubDomains
x-content-type-options: nosniff
x-frame-options: DENY
referrer-policy: strict-origin-when-cross-origin
content-security-policy: default-src 'self'; ...
```

### Step 3 — Next.js next.config.js

`frontend/next.config.js` 또는 `next.config.mjs` 에 추가:

```js
// CP240 — HTTP 보안 헤더 5종.
// CSP 는 lightweight-charts 호환 위해 'unsafe-inline' / 'unsafe-eval' 임시 허용.
// 향후 nonce 적용 (Next.js 14 middleware-based nonce) 가능 → ADR-0031.

const securityHeaders = [
  {
    key: 'Strict-Transport-Security',
    value: 'max-age=63072000; includeSubDomains',
  },
  {
    key: 'X-Content-Type-Options',
    value: 'nosniff',
  },
  {
    key: 'X-Frame-Options',
    value: 'DENY',
  },
  {
    key: 'Referrer-Policy',
    value: 'strict-origin-when-cross-origin',
  },
  {
    key: 'Content-Security-Policy',
    value: [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",   // lightweight-charts 호환
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "connect-src 'self' https://lens-backend-7stj.onrender.com",
      "frame-ancestors 'none'",
      "base-uri 'self'",
      "form-action 'self'",
    ].join('; '),
  },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  // ... 기존 설정 유지 ...
  async headers() {
    return [
      {
        source: '/:path*',
        headers: securityHeaders,
      },
    ];
  },
};

module.exports = nextConfig;
```

만약 `next.config.mjs` 사용 중이면 `export default nextConfig` 형식으로.

### Step 4 — 로컬 frontend 검증

```powershell
cd C:\Users\user\lens\frontend
npm run dev
# 브라우저 http://localhost:3000 접속
# DevTools → Network 탭 → 어떤 응답 클릭 → Headers → Response Headers
# 5개 헤더 모두 보이는지 확인
```

추가로:
```powershell
curl -I http://localhost:3000
```

### Step 5 — CSP 조정 (lightweight-charts 호환)

문제 가능성:
1. lightweight-charts 가 `Function` constructor 사용 → `script-src 'unsafe-eval'` 필요
2. inline `<style>` 사용 → `style-src 'unsafe-inline'` 필요
3. inline `<script>` (Next.js hydration) → `script-src 'unsafe-inline'` 필요

→ Step 3 의 CSP 에 이미 `'unsafe-inline'` + `'unsafe-eval'` 박아둠. 시작 baseline.

**점진 좁히기** (선택, 시간 여유 시):
- Next.js nonce 미들웨어 적용 → `'unsafe-inline'` 제거
- lightweight-charts 가 `'unsafe-eval'` 정말 필요한지 라이브러리 코드 grep
- 적용 후 매번 Playwright smoke 재실행

**v1 권장**: `'unsafe-inline'` + `'unsafe-eval'` 유지 (A+ 등급 못 가지만 A 등급 충분). ADR-0031 에 trade-off 명시. v2 에서 nonce 도입 결정.

**CSP 위반 모니터링** (선택):
`report-uri` 추가:
```
"content-security-policy: ...; report-uri /api/v1/csp-report"
```
→ backend 에 `/api/v1/csp-report` endpoint 추가하면 violation 로그 수집 가능. v1 보류, v2 도입.

### Step 6 — Playwright smoke 통과

```powershell
cd C:\Users\user\lens\frontend
npm run test:e2e
```

기존 smoke 가 차트 element 확인 안 하면 한 줄 추가:

`tests/e2e/main_flow.spec.ts` (또는 동일 파일) 에:
```ts
test('chart renders under CSP', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('[data-testid="price-chart"]', { state: 'visible' });
  // CSP violation 있으면 console error → fail
  const errors: string[] = [];
  page.on('console', msg => {
    if (msg.type() === 'error' && msg.text().includes('Content Security Policy')) {
      errors.push(msg.text());
    }
  });
  await page.waitForTimeout(2000);
  expect(errors).toEqual([]);
});
```

(차트 selector 가 다르면 실제 selector 로 교체.)

### Step 7 — Production 배포

```powershell
git add backend/app/core/security_headers.py backend/app/main.py frontend/next.config.js
git commit -m "CP240 Step 1-3: security headers (backend middleware + next.config)"
git push origin main
```

- Render: backend redeploy 자동 트리거 (~3분)
- Vercel: frontend redeploy 자동 트리거 (~2분)

### Step 8 — securityheaders.com 검증

브라우저:
1. https://securityheaders.com 접속
2. 입력란에 production URL: `https://lens-ten-delta.vercel.app`
3. Scan
4. 등급 확인 — **A 이상 목표**

기대:
- A+: nonce + CSP strict + `'unsafe-inline'` 없음 (v1 도달 어려움)
- **A**: 5 헤더 다 박힘, `'unsafe-inline'` 임시 허용 (v1 목표)
- B: 일부 빠짐
- F: 베이스라인

backend URL 도 별도 검사:
- `https://lens-backend-7stj.onrender.com`

### Step 9 — Mozilla Observatory (second opinion)

브라우저:
1. https://observatory.mozilla.org
2. 동일 URL 입력 → Scan
3. 점수 확인. Lens 같은 SPA 는 90+ 목표 어려움 (cookie / subresource integrity 같은 항목 영향). 75+ 면 OK.

### Step 10 — 보고서 + ADR

`docs/cp240_security_headers_report.md`:

```markdown
# CP240 Security Headers 보고서

## 결과
- 5 헤더 박힘 (backend middleware + next.config.js)
- securityheaders.com: A (frontend), A (backend)
- Mozilla Observatory: <점수>
- Playwright smoke: PASS (차트 렌더링 + CSP violation 0)

## CSP 정책
- script-src: 'self' 'unsafe-inline' 'unsafe-eval' (lightweight-charts 호환, v2 에서 nonce 도입 결정)
- style-src: 'self' 'unsafe-inline' (Next.js styled-jsx 호환)
- connect-src: 'self' https://lens-backend-7stj.onrender.com
- frame-ancestors: 'none' (clickjacking 차단)

## Trade-off
- 'unsafe-inline' 임시 허용 → A+ 못 감, A 등급 머무름. v2 에서 nonce 적용해 A+ 도전.

## 산출물
- backend/app/core/security_headers.py
- backend/app/main.py diff (middleware add)
- frontend/next.config.js diff (headers())
- docs/adr/0031_security_headers_csp.md
- (선택) tests/e2e 의 CSP 검증 테스트
```

ADR 양식은 §9 참조.

---

## 5. 회귀 안전망

- **CP223 BE snapshot**: response body 안 건드림 → snapshot 영향 0
- **CP230 FE smoke**: Playwright 가 차트 렌더링 + CSP violation 0 보장
- Step 6 CSP violation 모니터링 (console error) 핵심

---

## 6. 성공 기준 (L8)

- securityheaders.com: **A 이상**
- 5 헤더 모두 response 에 포함 (curl 확인)
- Playwright smoke PASS (차트 렌더링 + CSP violation 0)
- Lighthouse Best Practices 점수 향상 (baseline 비교)
- CI workflow PASS

---

## 7. 인터페이스 보존 (L7)

- response body 안 건드림 (header 만 추가)
- API contract 영향 0
- frontend / backend 응답 schema 동일
- **다만 CSP `connect-src` 에 Render URL 박지 않으면 frontend 가 backend 호출 못 함** → 이게 깨지면 즉시 발견

---

## 8. Lens 특화 (L9)

- **`connect-src` 에 `https://lens-backend-7stj.onrender.com` 박기 필수** — Vercel + Render cross-origin
- lightweight-charts 가 `'unsafe-eval'` 필요한지 점검 (라이브러리 코드)
- Next.js 14 hydration 이 inline script 많이 씀 → `'unsafe-inline'` 유지
- Vercel preview 도메인 (`https://lens-*.vercel.app`) 에도 자동 적용 (next.config.js 가 모든 환경 공통)
- `/api/v1/health/live` 같은 가벼운 endpoint 까지 미들웨어 거침 → 성능 영향 미미 (헤더 5줄)

---

## 9. ADR-0031 작성 가이드

파일: `docs/adr/0031_security_headers_csp.md`

```markdown
# ADR-0031: HTTP Security Headers + CSP Policy

## Status
Accepted (2026-06-06)

## Context
보안 헤더 0 → securityheaders.com F. OWASP A05 무방비. v2 Auth 진입 전 baseline 박아야.

## Decision
1. **5 헤더 박음** (HSTS / X-Content-Type-Options / X-Frame-Options / Referrer-Policy / CSP).
2. **HSTS max-age = 2년** (63072000s) + `includeSubDomains`. preload 등록은 별도 결정.
3. **X-Frame-Options: DENY** + CSP `frame-ancestors 'none'` 이중 보장 (구형 브라우저 호환).
4. **CSP 정책**:
   - `default-src 'self'` (외부 리소스 차단)
   - `script-src 'self' 'unsafe-inline' 'unsafe-eval'` — lightweight-charts + Next.js hydration 호환. **임시 허용**, v2 에서 nonce 적용해 제거 검토.
   - `connect-src 'self' https://lens-backend-7stj.onrender.com` — Vercel → Render cross-origin 허용.
   - `style-src 'self' 'unsafe-inline'` — Next.js styled-jsx 호환.
5. **Backend + Frontend 양쪽**: defense in depth. API 응답에도 박지만 효과는 frontend 가 주.
6. **보고**: CSP violation report 수집 (report-uri) 은 v2 도입.

## Consequences
- securityheaders.com A 등급 (A+ 가려면 'unsafe-inline' 제거 필요 → v2)
- lightweight-charts 가 CSP 위반하면 차트 안 그려짐 → Playwright smoke 가 회귀 감지
- HSTS preload list 등록 X (cancel 어려움 → 신중. v2 결정)
- `'unsafe-eval'` 허용은 XSS 방어 약화. 다만 v1 사용자 입력 거의 없어 위험 작음 (ticker 만 입력)

## References
- OWASP Top 10 A05
- MDN Web Security Headers
- web.dev/csp
```

---

## 10. 자동 실행 적합도

| Step | 자동 | 사람 확인 |
|---|---|---|
| 1 | ✅ | — |
| 2 | ✅ | — |
| 3 | ✅ | — |
| 4 | △ | 브라우저 DevTools 사용자 확인 |
| 5 | △ | CSP 조정 결정 (시각 회귀 점검) |
| 6 | ✅ | — |
| 7 | ✅ | git push 자동 trigger |
| 8 | ❌ | **securityheaders.com 결과 사용자 확인** |
| 9 | ❌ | **Mozilla Observatory 사용자 확인** |
| 10 | ✅ | — |

---

## 11. 종료 후 commit / 보고

### 권장 commit 분할

```
CP240 Step 1: backend SecurityHeadersMiddleware (new file)
CP240 Step 2: wire SecurityHeadersMiddleware in main.py
CP240 Step 3: next.config.js security headers
CP240 Step 5: CSP narrowing (lightweight-charts compat verified)
CP240 Step 6: e2e CSP violation guard test
CP240 report + ADR-0031 (security headers + CSP)
```

### 보고서
`docs/cp240_security_headers_report.md` (securityheaders.com 등급 스크린샷 첨부 권장)

### ADR
`docs/adr/0031_security_headers_csp.md`

---

**진입 조건**: CP223 BE snapshot + CP230 FE smoke (회귀 안전망 필수).
**다음 CP**: CP241 (Input validation).
**리스크**: Step 5 CSP 너무 strict 하면 차트 깨짐 → Step 6 Playwright 가 즉시 감지.
