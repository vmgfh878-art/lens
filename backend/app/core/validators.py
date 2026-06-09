"""CP241 — User input validation primitives.

모든 user input (path / query) 에 형식 검증 박음.

OWASP A03 Injection 예방:
- Path traversal (`../`, `%2e%2e`)
- Null byte (`\\x00`, `%00`)
- SQL fragments (`'; --`, `OR 1=1`)
- 비정상 형식 → 내부 에러 → 정보 노출

ticker 패턴:
- 미국: AAPL / MSFT / GOOGL / BRK.B / BF.B (점 표기) / BRK-B / BF-B (yfinance
  하이픈 표기) — 1~5자 + 옵션 `[.-]X` 1~2자. 운영 ticker_id_map 은 yfinance
  의 `-` 표기 사용 (CP241 Step 1 dry 검증으로 확인).
- 한국: 005930.KS / 035720.KQ (6자리 숫자 + .KS/.KQ)

레퍼런스: Pydantic v2 StringConstraints + FastAPI Annotated path/query.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import StringConstraints

# 미국: 대문자 1~5 + 옵션 (`[.-]대문자 1~2`). 점 또는 하이픈 둘 다 허용
# (yfinance 표기 `BRK-B`, Yahoo `BRK.B` 모두 통과).
# 한국: 6자리 숫자 + `.KS` 또는 `.KQ`.
# 통합 패턴 (regex alternation):
TICKER_PATTERN = r"^([A-Z]{1,5}(?:[.-][A-Z]{1,2})?|\d{6}\.K[SQ])$"

# Pydantic Annotated type — FastAPI 의 path/query param 에 그대로 박을 수 있음.
# FastAPI 가 자동으로 422 응답 (전역 handler 가 400 + 통일 schema 로 변환 — Step 5).
TickerStr = Annotated[
    str,
    StringConstraints(
        pattern=TICKER_PATTERN,
        min_length=1,
        max_length=10,
        strip_whitespace=True,
    ),
]


# 검색 query: 영문 + 숫자 + 점 + 공백 만. 특수문자 (`'`, `;`, `<`, `>`, `=`, ...) 차단.
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


# 타임프레임: enum 강제. 라우터에서 `Literal["1D", "1W", "1M"]` 직접 사용 권장.


_TICKER_RE = re.compile(TICKER_PATTERN)


def is_valid_ticker(value: str) -> bool:
    """Pydantic 외부에서 ticker 검증 헬퍼.

    예: service layer 의 추가 보호, daily refresh dry 검증 (CP241 Step 7),
    ticker_id_map 전수 통과 검증 등 Pydantic 우회 경로 대비.

    Args:
        value: 검증 대상 문자열.

    Returns:
        valid 면 True. None / non-str / 패턴 불일치 면 False.
    """
    if not isinstance(value, str):
        return False
    return bool(_TICKER_RE.match(value.strip()))
