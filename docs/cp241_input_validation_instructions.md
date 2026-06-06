# CP241 지시서 — Input Validation 강화 (Pydantic constr) (OWASP A03)

> 작성: 2026-06-06. 트랙: Lens 보안 트랙. ADR-0032 동반.
> 사용자 환경: Windows PowerShell. 한국어 보고서.

---

## 0. 한 줄 목표

모든 user input (path param `ticker`, query param `timeframe`/`days`/`horizon`/`search`)에 Pydantic constraint 적용. Path traversal / null byte / SQL injection / 비정상 입력 차단. 정상 ticker (S&P 500 + 한국 종목 포맷) 회귀 0.

---

## 1. 진단

| 항목 | 현재 |
|---|---|
| `{ticker}` path param 검증 | Type 만 (str) — **형식 검증 0** |
| query param 명시적 constraint | 일부 Pydantic 기본 type 만 |
| Path traversal 차단 | 없음 (parquet 경로는 server hardcoded 이라 위험 작지만 defense in depth) |
| SQL injection 차단 | SQLAlchemy ORM 이 자동 (CP236b) — 다만 user input 직접 string concat 가능성 점검 필요 |
| Null byte / 특수문자 차단 | 없음 |
| 검증 실패 응답 schema | main.py 전역 핸들러 표준 있음 — 통일 확인 필요 |

**위험 예시**:
- `/api/v1/predictions/line/../../etc/passwd` (path traversal)
- `/api/v1/predictions/line/AAPL%00.txt` (null byte injection)
- `/api/v1/stocks?search=' OR 1=1--` (SQL fragment, ORM 이라 안전하지만 정직)
- `/api/v1/predictions/line/lowercaseticker` (예상 외 형식 → 내부 에러 → 정보 노출)

**OWASP**: A03 Injection (예방).

---

## 2. 변경 내용

- `backend/app/core/validators.py` 신규 — `TickerStr` Pydantic constraint + 보조 validator
- `routers/v1/predictions.py`, `routers/v1/stocks.py` 등 ticker path param 을 `TickerStr` 로 교체
- query param 에 명시적 `Query(..., ge=, le=, max_length=)` constraint
- search query 에 max_length + 패턴 제한
- 검증 실패 시 통일된 error schema 응답 확인
- pytest negative test 추가 (6개+ invalid input 케이스)
- Playwright smoke 통과 (정상 ticker 회귀 0)
- ADR-0032 작성

---

## 3. Step 분할

| Step | 내용 | 위험 | 시간 | 자동/수동 |
|---|---|---|---|---|
| 1 | `backend/app/core/validators.py` 신규 (`TickerStr` + Korean ticker 패턴 + 보조) | 매우낮음 (additive) | 30분 | 자동 |
| 2 | 모든 `{ticker}` path param 라우터를 `TickerStr` 로 교체 (`predictions.py`, `stocks.py` 등) | 낮음 (signature 변경 없음, 검증만 추가) | 30분 | 자동 |
| 3 | query param 명시적 `Query(...)` constraint (timeframe / days / horizon / limit / search) | 매우낮음 (additive) | 30분 | 자동 |
| 4 | search query (`?search=AA`) max_length + 패턴 제한 | 낮음 | 15분 | 자동 |
| 5 | 검증 실패 응답 schema 통일 확인 — `{"error": {"code": "INVALID_TICKER", ...}, "meta": {"request_id": "..."}}` | 낮음 | 15분 | 자동 |
| 6 | pytest negative test 추가 (path traversal / SQL fragment / null byte / 빈 / 길이 초과 / lowercase) | 낮음 | 1h | 자동 |
| 7 | Playwright smoke 재확인 — 정상 ticker (AAPL, MSFT, BRK.B) 회귀 0 | 매우낮음 | 5분 | 자동 |
| 8 | `docs/cp241_input_validation_report.md` + ADR-0032 | 매우낮음 | 30분 | 자동 |

---

## 4. 각 Step 정확한 명령 / 코드

### Step 1 — validators.py 신규

`backend/app/core/validators.py` (신규):

```python
"""CP241 — User input validation primitives.

모든 user input (path / query) 에 형식 검증 박음.

OWASP A03 Injection 예방:
- Path traversal (../, %2e%2e)
- Null byte (\\x00, %00)
- SQL fragments ('; --, OR 1=1)
- 비정상 형식 → 내부 에러 → 정보 노출

ticker 패턴:
- 미국: AAPL, MSFT, BRK.B (1~5자 + 옵션 .X)
- 한국: 005930.KS, 005930.KQ (6자리 숫자 + .KS/.KQ)

레퍼런스: Pydantic v2 constr / StringConstraints.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import StringConstraints

# 미국 ticker: 대문자 1~5 + 옵션 (.대문자 1~2)
# 한국 ticker: 6자리 숫자 + .KS / .KQ
TICKER_PATTERN = r"^([A-Z]{1,5}(?:\.[A-Z]{1,2})?|\d{6}\.K[SQ])$"

# Pydantic Annotated type — path param 에 그대로 박을 수 있음.
TickerStr = Annotated[
    str,
    StringConstraints(
        pattern=TICKER_PATTERN,
        min_length=1,
        max_length=10,
        strip_whitespace=True,
    ),
]


# 검색 query: 영문 + 숫자 + 점 + 공백 만 (특수문자 차단).
SEARCH_PATTERN = r"^[A-Za-z0-9. ]{0,40}$"

SearchStr = Annotated[
    str,
    StringConstraints(
        pattern=SEARCH_PATTERN,
        min_length=0,
        max_length=40,
        strip_whitespace=True,
    ),
]


# 타임프레임: 명시적 enum (1D / 1W / 1M)
# 라우터에서 Literal["1D", "1W", "1M"] 직접 사용 권장.


_TICKER_RE = re.compile(TICKER_PATTERN)


def is_valid_ticker(value: str) -> bool:
    """Pydantic 외부에서 ticker 검증 헬퍼 (예: service layer 추가 보호)."""
    if not isinstance(value, str):
        return False
    return bool(_TICKER_RE.match(value.strip()))
```

### Step 2 — 라우터에 적용

대상 파일들 (grep 으로 사전 확인):
```powershell
cd C:\Users\user\lens
grep -rn "{ticker}" backend/app/routers/
```

예상 위치 (현재 코드 기준):
- `backend/app/routers/v1/predictions.py` — `/predictions/line/{ticker}` 등 5개+
- `backend/app/routers/v1/stocks.py` — `/stocks/{ticker}/prices` 등

각 endpoint 수정 예:

**Before**:
```python
@router.get("/line/{ticker}")
def get_line_predictions(ticker: str, days: int = 365):
    ...
```

**After**:
```python
from app.core.validators import TickerStr

@router.get("/line/{ticker}")
def get_line_predictions(
    ticker: TickerStr,
    days: Annotated[int, Query(ge=1, le=3650)] = 365,
):
    ...
```

(`Annotated` import 는 `typing` 또는 `typing_extensions` 에서.)

### Step 3 — query param constraint 일괄

예시:
```python
from fastapi import Query
from typing import Annotated, Literal

@router.get("/band/1d/{ticker}")
def get_band_1d(
    ticker: TickerStr,
    days: Annotated[int, Query(ge=1, le=3650)] = 365,
    horizon: Annotated[int, Query(ge=1, le=30)] = 5,
):
    ...

@router.get("/band/1w/{ticker}")
def get_band_1w(
    ticker: TickerStr,
    days: Annotated[int, Query(ge=1, le=3650)] = 730,
    horizon: Annotated[int, Query(ge=1, le=10)] = 4,
):
    ...

@router.get("/prices")
def get_prices(
    ticker: TickerStr,
    timeframe: Literal["1D", "1W", "1M"] = "1D",
    limit: Annotated[int, Query(ge=1, le=3000)] = 300,
):
    ...
```

`limit` 상한 (예: 3000) 은 응답 크기 폭증 차단 = DoS 예방.

### Step 4 — search query

```python
from app.core.validators import SearchStr

@router.get("/stocks")
def search_stocks(
    search: SearchStr = "",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    ...
```

검색 입력은 frontend 가 사용자 타이핑 그대로 보내니 max_length + 패턴 제한이 핵심.

### Step 5 — 검증 실패 응답 schema

FastAPI 가 Pydantic validation 실패 시 기본 422 응답. main.py 에 이미 전역 exception handler 가 있는지 확인:

```powershell
grep -n "RequestValidationError\|exception_handler" backend/app/main.py
```

만약 표준 schema (`{error: {code, message}, meta: {request_id}}`) 로 변환하는 핸들러 없으면 추가:

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "INVALID_INPUT",
                "message": "Request validation failed",
                "details": [
                    {"loc": err["loc"], "type": err["type"]}
                    for err in exc.errors()
                ],
            },
            "meta": {"request_id": getattr(request.state, "request_id", None)},
        },
    )
```

(상세 메시지는 `details` 에 minimal — 공격자에게 "이 필드는 이런 형식" 힌트 주지 않게. type 정도만.)

### Step 6 — pytest negative tests

`backend/tests/test_cp241_input_validation.py` 신규:

```python
"""CP241 — Input validation negative tests.

invalid input 6+ 케이스에 400 반환 + 정상 입력 200 회귀.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize(
    "invalid_ticker",
    [
        "../../etc/passwd",        # path traversal
        "AAPL/../",                # 슬래시 포함
        "AAPL\x00.txt",            # null byte
        "'; DROP TABLE users;--",  # SQL fragment
        "<script>alert(1)</script>",  # XSS payload
        "aapl",                    # lowercase
        "",                        # 빈 문자열
        "TOOLONGTICKER123",        # 길이 초과
        "AA PL",                   # 공백 포함
        "AAPL$",                   # 특수문자
    ],
)
def test_invalid_ticker_rejected(invalid_ticker):
    response = client.get(f"/api/v1/predictions/line/{invalid_ticker}")
    assert response.status_code in (400, 404, 422), (
        f"invalid ticker {invalid_ticker!r} got {response.status_code}"
    )


@pytest.mark.parametrize(
    "valid_ticker",
    [
        "AAPL",       # 표준 미국
        "MSFT",
        "BRK.B",      # 점 포함 미국
        "005930.KS",  # 한국 코스피
        "035720.KQ",  # 한국 코스닥
        "A",          # 1자
        "GOOGL",      # 5자
    ],
)
def test_valid_ticker_accepted(valid_ticker):
    response = client.get(f"/api/v1/predictions/line/{valid_ticker}")
    # 200 (데이터 있음) 또는 404 (데이터 없음)
    # 422/400 (검증 실패) 아니어야 함
    assert response.status_code in (200, 404), (
        f"valid ticker {valid_ticker!r} got {response.status_code}"
    )


def test_query_limit_upper_bound():
    """limit 상한 초과 시 차단 (DoS 예방)."""
    response = client.get("/api/v1/stocks?limit=99999")
    assert response.status_code in (400, 422)


def test_search_max_length():
    """search 길이 초과 차단."""
    response = client.get("/api/v1/stocks?search=" + "A" * 100)
    assert response.status_code in (400, 422)


def test_search_special_chars_rejected():
    """search 특수문자 차단."""
    response = client.get("/api/v1/stocks?search=' OR 1=1--")
    assert response.status_code in (400, 422)
```

실행:
```powershell
cd C:\Users\user\lens
pytest backend/tests/test_cp241_input_validation.py -v
```

### Step 7 — Playwright smoke 재확인

```powershell
cd C:\Users\user\lens\frontend
npm run test:e2e
```

기존 smoke 가 AAPL 같은 정상 ticker 로 검색 → 차트 로드 확인. 통과해야 정상 입력 회귀 0 보장.

### Step 8 — 보고서 + ADR

`docs/cp241_input_validation_report.md`:

```markdown
# CP241 Input Validation 보고서

## 적용 범위
- ticker path param: TickerStr (8 endpoint)
- query param: timeframe Literal / days/horizon/limit ge,le / search SearchStr
- 검증 실패: 통일된 error schema 응답 (400)

## 적용 라우터
- routers/v1/predictions.py (line / band/1d / band/1w / latest / history)
- routers/v1/stocks.py (prices / indicators / search)
- (다른 라우터 추가 시 동일 패턴)

## ticker 패턴
- 미국: ^[A-Z]{1,5}(\\.[A-Z]{1,2})?$
- 한국: ^\\d{6}\\.K[SQ]$
- 통합: 위 §2.1 코드 참조

## negative test 결과
- invalid ticker 10 케이스: 모두 400/404 (정상 거부)
- valid ticker 7 케이스: 모두 200/404 (정상 통과, 데이터 유무 따라)
- query 한계 테스트: limit 9999 / search 100자 / SQL fragment 모두 거부

## 회귀
- pytest backend/tests: PASS (N tests)
- CP223 snapshot: 0 diff
- Playwright smoke (AAPL): PASS

## 산출물
- backend/app/core/validators.py (신규)
- backend/app/routers/v1/predictions.py diff
- backend/app/routers/v1/stocks.py diff
- backend/app/main.py diff (RequestValidationError handler)
- backend/tests/test_cp241_input_validation.py (신규)
- docs/adr/0032_input_validation_pattern.md
```

ADR 양식은 §9 참조.

---

## 5. 회귀 안전망

- **CP223 BE snapshot**: 정상 ticker (AAPL 등) 응답 schema 동일 → diff 0 보장
- **CP230 FE smoke**: 정상 ticker 검색 → 차트 로드 → CSP violation 0
- **Step 6 negative test**: 새 안전망 (regression detection)
- **Step 7 Playwright**: 정상 흐름 회귀

---

## 6. 성공 기준 (L8)

- 모든 invalid ticker 케이스 400/404 (실제 데이터 path 안 닿음)
- 모든 valid ticker (미국 + 한국) 200/404 (데이터 유무에 따라, 400 아님)
- `pytest test_cp241_input_validation.py` 모든 assertion PASS
- CP223 snapshot diff 0
- Playwright smoke PASS

---

## 7. 인터페이스 보존 (L7)

- 정상 ticker 응답 schema 동일
- 실패 응답 schema: `{"error": {...}, "meta": {...}}` 통일 (frontend 가 이미 이 schema 처리 가정)
- frontend `apiErrors.ts` (`classifyApiError`) 가 새 `INVALID_INPUT` code 처리하는지 확인 권장

---

## 8. Lens 특화 (L9)

- **한국 종목 포맷 (`005930.KS`, `035720.KQ`)** 통과 — 패턴에 명시
- **BRK.B 같은 점 포함 미국 ticker** 통과 — 패턴 `[A-Z]{1,5}(\.[A-Z]{1,2})?` 가 커버
- **ETF (SPY, QQQ, IWM)** 통과 — 일반 미국 ticker 패턴에 포함
- 운영 모델 3개 추론 영향 0 (정상 ticker 회귀)
- daily refresh 영향 0 (ticker_id_map 의 모든 ticker 가 패턴 통과 — Step 7 전에 1회 dry 검증 권장)

검증 명령:
```powershell
cd C:\Users\user\lens
python -c "
import json
from pathlib import Path
from backend.app.core.validators import is_valid_ticker
m = json.loads(Path('ai/cache/ticker_id_map_1d.json').read_text())
bad = [t for t in m['mapping'] if not is_valid_ticker(t)]
print(f'invalid {len(bad)}: {bad[:10]}')
"
```

→ 0 이어야 운영 ticker 전수 통과.

---

## 9. ADR-0032 작성 가이드

파일: `docs/adr/0032_input_validation_pattern.md`

```markdown
# ADR-0032: User Input Validation Pattern

## Status
Accepted (2026-06-06)

## Context
v1 의 모든 user input (path param `ticker`, query param) 에 형식 검증 0. SQLAlchemy ORM 이 SQL injection 자동 방어하지만 path traversal / null byte / 비정상 형식 차단 안 됨. OWASP A03 무방비.

## Decision
1. **Pydantic v2 StringConstraints 활용**: Annotated type 으로 path param 에 직접 적용. 별도 validator 함수 없이 FastAPI 가 자동 검증.
2. **ticker 패턴**: `^([A-Z]{1,5}(\\.[A-Z]{1,2})?|\\d{6}\\.K[SQ])$` — 미국 + 한국 종목 통합. ETF, BRK.B 같은 점 포함 ticker 자동 커버.
3. **query param**: `Query(ge=, le=, max_length=)` 명시. `limit` 상한 박아 DoS 예방.
4. **timeframe**: `Literal["1D", "1W", "1M"]` — enum 강제.
5. **search**: 영문/숫자/점/공백 만, 최대 40자.
6. **검증 실패 응답**: 통일 schema `{error: {code: "INVALID_INPUT", ...}, meta: {request_id}}`. 상세 메시지에 공격자 힌트 minimal.
7. **defense in depth**: service layer 에 `is_valid_ticker()` 보조 헬퍼 (Pydantic 우회 경로 대비).

## Consequences
- 정상 ticker (S&P 500 + 한국) 회귀 0 — 패턴 신중 설계
- 새 ticker 형식 (예: 미래 ETF 5자 초과) 추가 시 패턴 갱신 필요
- 검증 실패 응답 schema 변경 → frontend `apiErrors.ts` 의 `INVALID_INPUT` code 처리 추가
- pytest negative test 가 새 안전망 → 향후 라우터 추가 시 동일 패턴 강제

## References
- OWASP Top 10 A03
- Pydantic v2 StringConstraints
- FastAPI Query / Path docs
```

---

## 10. 자동 실행 적합도

| Step | 자동 | 사람 확인 |
|---|---|---|
| 1 | ✅ | — |
| 2 | ✅ | — |
| 3 | ✅ | — |
| 4 | ✅ | — |
| 5 | ✅ | — |
| 6 | ✅ | — |
| 7 | ✅ | — |
| 8 | ✅ | — |

→ **자동 적합도 매우 높음**. 패턴화되어 있음. agent 가 전수 가능.

---

## 11. 종료 후 commit / 보고

### 권장 commit 분할

```
CP241 Step 1: add core/validators.py (TickerStr + SearchStr)
CP241 Step 2: apply TickerStr to predictions.py routes
CP241 Step 3: apply TickerStr to stocks.py routes + query constraints
CP241 Step 4: apply SearchStr to search query
CP241 Step 5: unified RequestValidationError handler (main.py)
CP241 Step 6: pytest negative + valid ticker matrix tests
CP241 report + ADR-0032 (input validation pattern)
```

### 보고서
`docs/cp241_input_validation_report.md`

### ADR
`docs/adr/0032_input_validation_pattern.md`

---

**진입 조건**: CP223 BE snapshot 회귀 안전망 (정상 ticker 영향 0 보장).
**다음 CP**: CP242 (CORS + rate limit + 보안 트랙 종료).
**리스크**: 운영 ticker 중 패턴 안 맞는 게 있으면 daily refresh 깨짐 → §8 의 dry 검증 명령 필수 선행.
