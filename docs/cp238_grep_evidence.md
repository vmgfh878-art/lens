# CP238 Grep Evidence

> 작성: 2026-06-08. CP238 Step 6 frontend CVE justification 의 grep 근거 박제.
> 모든 명령은 Windows PowerShell 5.1 (cp949 환경). 검사 시점: 2026-06-08 ~01:30 KST.

## Backend (cf. docs/security/cve_exceptions.md §A 1-4)

### B1. starlette host header injection (CVE-2026-48710)

```powershell
Get-ChildItem -Path backend\app -Recurse -Filter *.py | Select-String -Pattern \
  "request\.url\.|request\.base_url|request\.headers\.get\(.host|X-Forwarded-Host|RedirectResponse|url_for\("
```

검사 대상: backend\app 46 *.py. **결과: 0 hit**.

### B2. starlette multipart sync block (CVE-2025-54121) + multipart-related

```powershell
Get-ChildItem -Path backend\app -Recurse -Filter *.py | Select-String -Pattern \
  "multipart|UploadFile|form_data|set_key|unset_key|dotenv\.set_key"
```

검사 대상: backend\app 46 *.py. **결과: 0 hit**.

### B3. starlette FileResponse Range DoS (CVE-2025-62727)

```powershell
Get-ChildItem -Path backend -Recurse -Filter *.py | Select-String -Pattern \
  "FileResponse|StaticFiles|StreamingResponse|send_file"
```

검사 대상: backend\* *.py. **결과: 0 hit**.

### B4. pyarrow Arrow C++ UAF (CVE-2026-25087)

```powershell
Get-ChildItem -Path . -Recurse -Filter *.py | Select-String -Pattern \
  "pyarrow|pa\.ipc|read_feather|IPC|Feather"
```

**결과: 6 파일 hit** — 모두 내부 fixture/학습 데이터:
- `scripts/reproduce_band_1w_cp178.py`
- `scripts/reproduce_band_1d_cp153.py`
- `scripts/reproduce_line_cp210.py`
- `ai/cp148_lm_1d_stage4_4f_failure_analysis.py`
- `ai/cp146_lm_eodhd500_line_full_training.py`
- `backend/db/scripts/export_parquet.py`

untrusted 외부 IPC/Feather 입력 0.

## Frontend (cf. docs/security/cve_exceptions.md §B 1-15)

### F1. Server Actions (`"use server"`)

```powershell
Select-String -Path frontend\src\**\*.ts,frontend\src\**\*.tsx \
  -Pattern '"use server"' -SimpleMatch
```

**결과: 0 hit**.

### F2. Pages Router (디렉토리 존재)

```powershell
Test-Path frontend\pages, frontend\src\pages
```

**결과: False, False** — Pages Router 미사용. App Router 만.

### F3. next/image import

```powershell
Select-String -Path frontend\src\**\*.ts,frontend\src\**\*.tsx \
  -Pattern "from\s+['\"]next/image['\"]"
```

**결과: 0 hit** — Image Optimization 미사용.

### F4. Middleware (파일 존재)

```powershell
Test-Path frontend\middleware.ts, frontend\src\middleware.ts
```

**결과: False, False** — Middleware 미사용.

### F5. WebSocket

```powershell
Select-String -Path frontend\src\**\*.ts,frontend\src\**\*.tsx \
  -Pattern "WebSocket|ws://|wss://"
```

**결과: 0 hit**.

### F6. i18n in next.config

```powershell
Select-String -Path frontend\next.config.* -Pattern "i18n"
```

**결과: 0 hit** — i18n 미사용.

### F7. App Router API routes (`src/app/api`)

```powershell
Get-ChildItem frontend\src\app\api -Recurse -ErrorAction SilentlyContinue
```

**결과: 디렉토리 없음** — App Router API route handler 미사용. `frontend/src/api/` 는
별개 (axios 기반 backend HTTP client 헬퍼: `baseClient.ts`, `client.ts`, `endpoints/`,
`types/`).

### F8. Vercel caching (ISR / revalidate / generateStaticParams / fetch next)

```powershell
Select-String -Path frontend\src\app\**\*.tsx,frontend\src\app\**\*.ts \
  -Pattern "revalidate|ISR|generateStaticParams|fetch\(.*\{.*next:"
```

**결과: 0 hit** — ISR / on-demand revalidate / generateStaticParams 미사용.
RSC cache poisoning 류 advisories 의 **mitigation factor** (cache 자체를 거의
안 쓰면 poisoning 영향 폭이 작아짐).

### F9. experimental cache / staleTimes in next.config

```powershell
Select-String -Path frontend\next.config.* -Pattern "experimental.*cache|staleTimes"
```

**결과: 0 hit** — next.config 의 cache 관련 experimental 옵션 미설정.

### F10. CSP nonces / Content-Security-Policy headers

```powershell
Select-String -Path frontend\next.config.* \
  -Pattern "headers\(\)|Content-Security-Policy|nonce"
Select-String -Path frontend\src\**\*.ts,frontend\src\**\*.tsx \
  -Pattern "Content-Security-Policy|cspNonce|nonce"
```

**결과: 0 hit** (config + src 둘 다) — CSP nonces 미사용.

### F11. next/script import + beforeInteractive strategy

```powershell
Select-String -Path frontend\src\**\*.ts,frontend\src\**\*.tsx \
  -Pattern "from\s+['\"]next/script['\"]"
Select-String -Path frontend\src\**\*.ts,frontend\src\**\*.tsx -Pattern "beforeInteractive"
```

**결과: 0 hit** (import + 키워드 둘 다) — beforeInteractive Script 미사용.

## next.config.mjs 실제 사용 패턴 (rewrites)

```javascript
const nextConfig = {
  async rewrites() {
    if (!proxyTarget) {
      return [];
    }
    return [
      {
        source: "/__backend/:path*",
        destination: `${proxyTarget}/:path*`,
      },
    ];
  },
};
```

**rewrites() 사용 → 영향 advisories**:
- GHSA-ggv3-7p47-pfv8 (HTTP request smuggling in rewrites)
- GHSA-3g8h-86w9-wvmq (Middleware/Proxy redirects cache poisoning — proxy 부분)

→ §B acknowledged.

## App Router + RSC default 사용 패턴

App Router (`src/app/layout.tsx`, `src/app/page.tsx`, `src/app/ClarityInit.tsx` 등)
의 default 는 **React Server Components**. `"use client"` directive 가 명시되지
않은 컴포넌트는 RSC 로 렌더링.

**RSC 사용 → 영향 advisories**:
- GHSA-h25m-26qc-wcjf (HTTP request deserialization DoS with insecure RSC)
- GHSA-q4gf-8mx6-v5v3 (DoS with Server Components)
- GHSA-8h8q-6873-q5fj (DoS with Server Components, 별개 ID)
- GHSA-vfv6-92ff-j949 (RSC cache-busting collision)
- GHSA-wfc6-r584-vfw7 (RSC cache poisoning)

→ §B acknowledged. ISR/revalidate 미사용 (F8/F9) 이 mitigation factor.

## 통계

- 검사 명령: 11 (backend 4 + frontend 11 — 일부 명령은 다중 패턴)
- 0 hit / 0 파일: 10
- 사용 패턴 발견: 1 (pyarrow 의 내부 데이터 read — untrusted 외부 X 라 결과적 영향 0)

frontend 의 실제 사용 패턴 (rewrites + RSC) 은 grep 외 별도 코드 read 로 식별.

## 재실행 절차

본 문서의 모든 grep 은 결정적. CP238 closure 후 회귀 검출 시 그대로 실행해
변경 여부 확인 가능. 새 endpoint / 새 사용 패턴 (multipart upload / file
serving / WebSocket / Image Optimization 등) 도입 시 즉시 본 문서 갱신 +
cve_exceptions.md 갱신.
