# ADR-0032: User Input Validation Pattern

Status: Accepted
Date: 2026-06-09
CP: CP241 (OWASP A03)

## Context

Lens v1 의 모든 user input (path param `{ticker}`, query param `timeframe` /
`days` / `horizon` / `limit` / `search`) 에 형식 검증 0. SQLAlchemy ORM 이 SQL
injection 자동 방어하지만 path traversal / null byte / 비정상 형식 차단 안 됨.
OWASP A03 Injection 무방비.

## Decision

### 1. Pydantic v2 `StringConstraints` 활용

`Annotated[str, StringConstraints(...)]` 으로 FastAPI path/query param 에
직접 적용. 별도 validator 함수 없이 FastAPI 가 자동 422 응답.

### 2. ticker 패턴

```
^([A-Z]{1,5}(?:[.-][A-Z]{1,2})?|\d{6}\.K[SQ])$
```

- 미국: AAPL / MSFT / GOOGL / BRK.B (점 표기) / BRK-B (yfinance 하이픈
  표기) — 1~5자 + 옵션 `[.-]X` 1~2자
- 한국: 005930.KS / 035720.KQ — 6자리 숫자 + `.KS` / `.KQ`
- ETF (SPY / QQQ / IWM) 자동 커버 (일반 미국 패턴)

**보강 (CP241 Step 1)**: 초기 `\.` (점만) 패턴 → 운영 ticker `BRK-B`, `BF-B`
2개 invalid 발견 → `[.-]` 으로 점/하이픈 둘 다 허용 → 1d 188 + 1w 421 모두
0 invalid 확보.

### 3. query param constraint

- **`timeframe`**: `Literal["1D", "1W"]` — Lens 운영 기준 (1M 미사용,
  CLAUDE.md sufficiency gate 의 1D/1W 만)
- **`limit`**: `Query(ge=1, le=<적정 상한>)` — 응답 크기 폭증 차단 (DoS 예방)
- **`days` / `horizon`**: `Query(ge=, le=)` — 의미적 한계 강제
- **`search`**: `SearchStr` (영문/숫자/점/공백, max 40자) — 특수문자 차단

### 4. 검증 실패 응답 schema

기존 `handle_validation_error` (main.py line 101) 활용 — 통일 schema
(`error.code`, `error.message`, `meta.request_id`) 유지.

**보강 (CP241 Step 5)**: `details` 를 `[{loc, type}]` 만 minimal 으로 변환.
공격자에게 raw pydantic `ctx.pattern` / input value 노출 차단.

### 5. defense in depth

- Pydantic constraint (1차 layer, FastAPI 자동 422)
- httpx client-side URL parser (0차 layer — null byte 등 RFC 3986 위반
  서버 도달 전 차단)
- `is_valid_ticker()` 헬퍼 (service layer 추가 보호, daily refresh dry 검증)

### 6. 영구 안전망

`backend/tests/test_input_validation.py` (22 test):
- invalid ticker 9 parametrize + null byte 1 → 차단
- valid ticker 7 parametrize → 통과
- query 한계 4 (limit 9999 / search 100자 / SQL fragment / timeframe 1M) → 차단
- 응답 schema 1 (통일 + minimal details) 검증

파일명 cp prefix 안 박음 (CP223 / CP240 와 일관, .gitignore `test_cp*.py` 우회).

## Consequences

### 장점
- OWASP A03 baseline 박힘 (path traversal / SQL fragment / 비정상 형식 차단)
- 정상 ticker (S&P 500 + 한국 + ETF + yfinance 하이픈 표기) 회귀 0 — 패턴
  보강으로 확인
- 검증 실패 응답 정보 노출 차단 (details minimal)
- 22 test 영구 안전망 → 향후 라우터 추가 시 동일 패턴 강제

### 단점 / Trade-off
- 새 ticker 형식 (예: 미래 ETF 5자 초과, 다른 거래소 .HK/.TO 등) 추가 시
  패턴 갱신 필요 — `is_valid_ticker` dry 검증으로 미리 감지
- frontend `apiErrors.ts` 가 422 `VALIDATION_ERROR` code 처리 가정 — schema
  보존 했으므로 영향 0

### Trade-off (acknowledged)
- `timeframe` Literal 1D/1W 만 — 미래 1M 도입 시 라우터 + 패턴 갱신 필수
- `roles` (product-history) 는 enum 강제 안 함 (지시서 명시 X) — 향후 보강
  여지

## v2 재검토

- 새 거래소 ticker 형식 (.HK, .TO, .DE 등) 도입 시 패턴 확장
- Auth 도입 시 사용자 input 다양화 → SearchStr 도 한국어 허용 검토
- `roles` 등 enum 강제 보강

## References

- OWASP Top 10 A03 (https://owasp.org/Top10/A03_2021-Injection/)
- Pydantic v2 StringConstraints (https://docs.pydantic.dev/latest/api/types/#pydantic.types.StringConstraints)
- FastAPI Path / Query (https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/)
- 본 트랙 보고서: `docs/cp241_input_validation_report.md`
