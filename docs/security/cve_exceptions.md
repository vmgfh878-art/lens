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

## References

- pip-audit `--ignore-vuln` 옵션 (https://github.com/pypa/pip-audit#configuration)
- OSV.dev advisory DB (https://osv.dev)
- 본 트랙 보고서: `docs/cp238_dependency_audit_report.md`
- 본 트랙 ADR: `docs/adr/0029_dependency_audit_policy.md`
