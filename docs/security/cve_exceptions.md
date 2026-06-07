# CVE Exceptions

> CP238 (2026-06-07) 기준. 4 CVE 가 pip-audit 에서 잔여 — 모두 Lens 사용 패턴상
> 실제 영향 0 확인 후 CI 게이트에서 `--ignore-vuln` 으로 제외. 재검토 주기 6개월
> (다음 검토: 2026-12-07). 패키지 bump 또는 CVE 의 fix 가능 시점 도래 시 즉시 갱신.

## 잔여 CVE 목록

### 1. pyarrow PYSEC-2026-113 (CVE-2026-25087)

- 패키지: `pyarrow==17.0.0`
- 공격 벡터: Apache Arrow C++ 의 Use-After-Free. IPC file (스트림 X) 을
  pre-buffering 활성 상태로 read 할 때, variadic buffer 가 포함된 IPC 면
  trigger 가능.
- Lens 사용 패턴:
  ```powershell
  Select-String -Path **/*.py -Pattern "pyarrow|pa\.ipc|read_feather|IPC|Feather"
  ```
  결과 6 파일 (scripts/reproduce_*.py 3개, ai/cp148_*.py + ai/cp146_*.py,
  backend/db/scripts/export_parquet.py). 모두 **내부 fixture/학습 데이터**.
  외부 untrusted IPC/Feather 입력 0.
- 영향도 결론: **0** (untrusted source 부재). 외부 사용자 input 으로 pyarrow IPC
  read 하는 endpoint 없음.
- Fix 가능 시점: pyarrow 23.0.1 — 다만 major bump (17 → 23) breaking 가능성 큼.
  pandas 호환 확인 후 별도 CP.

### 2. starlette PYSEC-2026-161 (CVE-2026-48710, GHSA-86qp-5c8j-p5mr)

- 패키지: `starlette==0.41.3` (fastapi 0.115.6 의 transitive)
- 공격 벡터: Host header 값 검증 없이 URL 재구성. attacker 가 host 부분에 path
  prepend 가능. routing 자체엔 영향 없지만 host 기반 redirect / url_for /
  notification 등에서 위험.
- Lens 사용 패턴:
  ```powershell
  Select-String -Path backend\app\**\*.py -Pattern \
    "request\.url\.|request\.base_url|request\.headers\.get\(.host|X-Forwarded-Host|RedirectResponse|url_for\("
  ```
  결과 **0 hit** (backend/app 46 *.py 검사).
- 영향도 결론: **0**. backend 가 host header 기반 URL 재구성을 어디서도 안 함.
- Fix 가능 시점: starlette 1.0.1 — fastapi 0.115.x 의 pin (`<0.39 → 0.41.x`)
  이라 fastapi major bump 필요. 별도 CP 에서 검토.

### 3. starlette GHSA-2c2j-9gv5-cj73 (CVE-2025-54121)

- 패키지: `starlette==0.41.3`
- 공격 벡터: multipart/form-data 의 큰 파일 (default max spool size 초과) 을
  파일 시스템으로 spill 할 때 sync I/O 가 main thread 차단 → CPU starvation.
- Lens 사용 패턴:
  ```powershell
  Select-String -Path backend -Pattern "multipart|UploadFile|form_data|set_key|unset_key|dotenv\.set_key"
  ```
  결과 **0 hit**.
- 영향도 결론: **0**. Lens API 는 multipart upload endpoint 없음. read-only
  predictions / scan / backtest 만 제공.
- Fix 가능 시점: starlette 0.47.2. fastapi pin 변경 필요.

### 4. starlette GHSA-7f5h-v6xp-fcq8 (CVE-2025-62727)

- 패키지: `starlette==0.41.3`
- 공격 벡터: `FileResponse` 의 Range header parsing/merging 이 quadratic
  time. attacker 가 crafted Range 보내 CPU 소진 가능.
- Lens 사용 패턴:
  ```powershell
  Select-String -Path backend -Pattern "FileResponse|StaticFiles|StreamingResponse|send_file"
  ```
  결과 **0 hit**. Lens 는 file serving / static asset 응답 안 함 (모두 JSON).
- 영향도 결론: **0**. FileResponse 미사용 + StaticFiles 미사용.
- Fix 가능 시점: starlette 0.49.1.

## 모니터링 / 갱신 정책

- **재검토 주기**: 6개월 (다음: 2026-12-07).
- **즉시 재검토 트리거**:
  - 새 endpoint 추가 시 (특히 multipart upload / file serving / host header
    기반 redirect) — 본 문서 사용 패턴 가정 무효화.
  - pip-audit 가 새 CVE 발견 시 — Lens 사용 패턴과 무관해도 본 문서 갱신.
  - fastapi 가 starlette 1.x 호환 minor 출시 시 — starlette 3개 CVE 모두 fix
    가능 → 본 문서 항목 2/3/4 제거.
- **갱신 절차**: 사용 패턴 grep 재실행 → 영향도 재확인 → 변경 사항 본 문서에
  date stamp 와 함께 기록.

## Frontend CVE (CP238 Step 6, 2026-06-08)

> next 14.2.35 + postcss devDep ^8 transitive. 15 advisories 잔여 (next 14
> + postcss 1) — 모두 audit-ci `allowlist` 로 박힘. Lens 사용 패턴 grep 근거는
> `docs/cp238_grep_evidence.md` 참조. 재검토 주기 **3개월** (표준 6개월 단축),
> v2 mitigation 트랙 = next 16 major bump (v2 Supabase Auth + production
> 본격화 시점 동시 진행).

### §A. Justified — Lens 사용 패턴 0 (실제 영향 0)

#### A1. Image Optimization 3 (next/image, remotePatterns, Optimization API)

- GHSA-9g9p-9gw9-jx7f (moderate) — self-hosted Image Optimizer remotePatterns DoS
- GHSA-3x4c-7xq6-9pq8 (moderate) — Unbounded next/image disk cache growth
- GHSA-h64f-5h5j-jqjh (moderate) — Image Optimization API DoS
- Lens grep: F3 (`next/image` import) → 0 hit. next.config.mjs 의 `images.remotePatterns` 미설정.
- 결론: **영향 0**. Image Optimization 자체를 안 씀.

#### A2. WebSocket SSRF (GHSA-c4j6-fc7j-m34r, high)

- 공격 벡터: WebSocket upgrade endpoint 가 attacker-controlled URL 로 SSRF.
- Lens grep: F5 (`WebSocket|ws://|wss://`) → 0 hit.
- 결론: **영향 0**. WebSocket endpoint 0.

#### A3. Pages Router i18n bypass (GHSA-36qx-fr4f-26g5, high)

- 공격 벡터: Pages Router + i18n 사용 시 Middleware/Proxy bypass.
- Lens grep: F2 (`pages/` dir) → False, F6 (next.config `i18n`) → 0 hit.
- 결론: **영향 0**. Pages Router 미사용 + i18n 미사용 (이중).

#### A4. CSP nonces XSS (GHSA-ffhc-5mcf-pf4q, moderate)

- 공격 벡터: App Router 가 CSP nonces 적용 시 nonce reuse / 우회 → XSS.
- Lens grep: F10 (next.config `headers()` + src 의 `Content-Security-Policy`/nonce) → 0 hit.
- 결론: **영향 0**. CSP nonce 자체를 안 박음.

#### A5. beforeInteractive Script XSS (GHSA-gx5p-jg67-6x7h, moderate)

- 공격 벡터: `next/script` 의 `strategy="beforeInteractive"` + untrusted input → XSS.
- Lens grep: F11 (`next/script` import + `beforeInteractive` 키워드) → 0 hit.
- 결론: **영향 0**. next/script 자체를 안 씀.

#### A6. postcss XSS (GHSA-qx2v-qp2m-jg93, moderate)

- 공격 벡터: PostCSS Stringify output 의 `</style>` unescaped → XSS (untrusted CSS
  input 을 build/render 할 때).
- Lens 사용 패턴: tailwindcss + autoprefixer 의 devDep. **build-time only**.
  runtime 에 사용자 input 으로 CSS stringify 안 함.
- 결론: **영향 0**. devDep 이 사용자 input 처리 0.

### §B. Acknowledged — Lens 사용 패턴 있음 (risk 인정, v2 mitigation)

#### B1. rewrites HTTP smuggling (GHSA-ggv3-7p47-pfv8, moderate)

- 공격 벡터: `next.config` 의 `rewrites()` 가 HTTP request smuggling 에 취약.
  attacker 가 chunked encoding 등으로 smuggled request 박음.
- Lens 사용 패턴: `next.config.mjs` 가 `/__backend/:path*` → `${proxyTarget}/:path*`
  same-origin proxy. **rewrites() 사용**.
- **최대 실제 영향**: vercel/render 경로의 cache poisoning 또는 backend 에 비정상
  request 도달. **금전/데이터 유출은 0** (backend Pydantic validation 거침).
- Mitigation factor:
  - Lens 는 인증/세션 없음 (v1) → smuggled request 가 다른 사용자 세션 hijack 불가
  - backend Pydantic input validation (CP241 예정) → invalid payload reject
  - 공개 데이터 (주가/predictions) 만 — 노출 시 손실 0
- v2 mitigation 트랙: next 16 major bump 동시 진행 (Supabase Auth 도입 시).

#### B2. Middleware/Proxy redirects cache poisoning (GHSA-3g8h-86w9-wvmq, low)

- 공격 벡터: Middleware 또는 Proxy redirects 에서 cache poisoning.
- Lens 사용 패턴: middleware 미사용 (F4 → False). 그러나 rewrites() 는 proxy 라
  proxy 부분에 영향 잠재.
- Mitigation factor: low severity + Lens 의 cache 가 거의 없음 (F8 ISR/revalidate
  0 hit, F9 staleTimes 0 hit) → cache poisoning 실효 매우 낮음.
- v2 mitigation: B1 와 동시.

#### B3-7. React Server Components 5

App Router 의 `src/app/layout.tsx`, `src/app/page.tsx`, `ClarityInit.tsx` 등은
`"use client"` directive 가 없는 한 default RSC.

- GHSA-h25m-26qc-wcjf (high) — HTTP request deserialization DoS with insecure RSC
- GHSA-q4gf-8mx6-v5v3 (high) — DoS with Server Components
- GHSA-8h8q-6873-q5fj (high) — DoS with Server Components (별개 ID)
- GHSA-vfv6-92ff-j949 (low) — RSC cache-busting collision
- GHSA-wfc6-r584-vfw7 (moderate) — RSC cache poisoning in responses

- Lens 사용 패턴: App Router + RSC default (F1 Server Actions `"use server"` 0 hit
  이지만 RSC 자체는 grep 없이 사용).
- **최대 실제 영향**: DoS (vercel quota 초과). cache poisoning 류는 mitigation
  factor 적용 시 실효 낮음.
- Mitigation factor:
  - ISR/revalidate/generateStaticParams 0 hit (F8) → server-side cache 자체가
    거의 없음 → server cache poisoning / cache-busting collision 실효 매우 낮음
  - experimental.staleTimes 0 hit (F9) → cache 정책 default
  - browser cache poisoning (vfv6 RSC cache-busting collision 의 변형) 은
    사용자 본인 browser 에 잘못된 RSC response cache → 다음 페이지 로드 시
    잘못된 UI. 단 인증 없음 → cross-user impact 0, 데이터 유출 0
  - RSC DoS (h25m/q4gf/8h8q) 는 cache 와 무관 request-level trigger →
    Vercel function quota / 응답 시간 영향. backend (별 process) 까지는 X
  - 인증 없음 → cross-user impact 0
  - 공개 read-only 데이터만 → 유출 시 손실 0
- v2 mitigation: next 16 major bump (RSC 호환 검증 + React 19 호환).

### Acknowledgement (사용자 서명)

본 §B 의 7 acknowledged advisories 는 v1 운영 시점에 Lens 사용 패턴 mitigating
factor 와 함께 risk 인정하고 진행한다. v2 (Supabase Auth + production 본격화)
진입 시 next 16 major bump 동시 진행을 commitment 으로 박는다.

```
Acknowledged by: 김지형
Date: 2026-06-08
v2 mitigation commitment: Supabase Auth + production 본격화 트랙 진입 시
  next 16 major bump 동시 진행 (RSC + React 19 호환 재검증 포함).
Pre-sign review (agent 세 번 검수): rewrites 영향 (vercel/render quota DoS +
  log injection 가능, 금전/데이터 유출 0) / RSC 영향 (vercel function quota
  DoS + browser cache poisoning, cross-user 0 + 데이터 유출 0) / mitigation
  factor 정확성 (인증 없음 / 공개 데이터 / ISR 미사용 / CP241 예정) 모두
  정직성 확인. browser cache 영향 1건 본 acknowledgement 직전 보강 박힘.
```

### Frontend 모니터링 / 갱신 정책

- **재검토 주기**: 3개월 (다음: 2026-09-08). backend 6개월보다 짧음 — frontend
  advisories 가 자주 갱신되고 acknowledged 7 의 risk acknowledged 이라 빠른
  주기 필요.
- **즉시 재검토 트리거**:
  - 새 Next.js feature 도입 (next/image / WebSocket / Middleware / CSP nonces /
    next/script / ISR / Server Actions) — `docs/cp238_grep_evidence.md` 의 grep
    재실행 후 §A → §B 재분류
  - next 14.x 의 new advisory 출시 시
  - 사용자 input 받는 endpoint (예: 검색 / 댓글 / 업로드) 도입 시 — Pydantic
    validation 만으로 충분한지 재검토
  - v2 (Auth) 진입 시 — next 16 major bump 동시
- **갱신 절차**: `docs/cp238_grep_evidence.md` 재실행 → 결과 차이 박제 → 본 문서
  §A/§B 갱신 → `frontend/audit-ci.json` allowlist 갱신.

## References

- pip-audit `--ignore-vuln` 옵션 (https://github.com/pypa/pip-audit#configuration)
- audit-ci (https://github.com/IBM/audit-ci)
- OSV.dev advisory DB (https://osv.dev)
- GitHub Advisory DB (https://github.com/advisories)
- Lens grep 근거: `docs/cp238_grep_evidence.md`
- 본 트랙 보고서: `docs/cp238_dependency_audit_report.md`
- 본 트랙 ADR: `docs/adr/0029_dependency_audit_policy.md`
