"""CP241 — Input validation negative + valid ticker tests.

invalid input 10+ 케이스 → 422 (FastAPI 의 pydantic validation).
valid ticker (S&P 500 + 한국 + ETF) 7 케이스 → 200/404 (데이터 유무에 따라).

파일명에 cp prefix 안 박은 이유: 영구 안전망 (.gitignore 의 test_cp*.py
우회, CP223 test_characterization_api.py 와 동일 정책).
"""

from __future__ import annotations

import pytest

from tests.conftest import FIXED_HEADERS


@pytest.mark.parametrize(
    "invalid_ticker",
    [
        "../../etc/passwd",  # path traversal
        "AAPL..",  # 슬래시 변형
        "'OR1=1--",  # SQL fragment
        "<scr>",  # XSS payload
        "aapl",  # lowercase
        "TOOLONG123",  # 길이 초과 (10자 + 패턴 불일치)
        "AA PL",  # 공백 포함
        "AAPL$",  # 특수문자
        "AA_PL",  # 언더스코어
    ],
)
def test_invalid_ticker_rejected_on_predictions(client, invalid_ticker):
    """predictions endpoint 에 invalid ticker → 차단 (data path 안 닿음).

    차단 status:
    - 422: Pydantic constraint 위반 (대부분의 케이스)
    - 404: FastAPI routing 단계 unmatch (path traversal `../` — routing 이 먼저 reject)
    - 400: 일부 malformed input
    """
    response = client.get(
        f"/api/v1/predictions/line/{invalid_ticker}",
        headers=FIXED_HEADERS,
    )
    assert response.status_code in (400, 404, 422), (
        f"invalid ticker {invalid_ticker!r} got {response.status_code} "
        f"(expected one of 400/404/422): {response.text[:200]}"
    )


def test_null_byte_rejected_by_client_url_parser(client):
    """null byte 박힌 URL → httpx client-side InvalidURL 즉시 차단.

    서버 도달 전 차단 (가장 강한 layer). production 의 nginx/render proxy
    도 동일하게 차단 — RFC 3986 URL 표준 위반.
    """
    from httpx import InvalidURL

    with pytest.raises(InvalidURL):
        client.get(
            "/api/v1/predictions/line/AAPL\x00",
            headers=FIXED_HEADERS,
        )


@pytest.mark.parametrize(
    "valid_ticker",
    [
        "AAPL",  # 표준 미국
        "MSFT",
        "GOOGL",  # 5자
        "BRK-B",  # 하이픈 (yfinance 표기, 운영 ticker)
        "BF-B",  # 동일
        "005930.KS",  # 한국 코스피
        "035720.KQ",  # 한국 코스닥
    ],
)
def test_valid_ticker_accepted_on_predictions(client, valid_ticker):
    """valid ticker → 200 (데이터 있음) 또는 404 (데이터 없음). 422 (검증 실패) 아님."""
    response = client.get(
        f"/api/v1/predictions/line/{valid_ticker}",
        headers=FIXED_HEADERS,
    )
    assert response.status_code in (200, 404), (
        f"valid ticker {valid_ticker!r} got {response.status_code}: " f"{response.text[:200]}"
    )


def test_query_limit_upper_bound_rejected(client):
    """limit 상한 초과 → 422 (DoS 예방)."""
    response = client.get(
        "/api/v1/stocks",
        params={"limit": 99999},
        headers=FIXED_HEADERS,
    )
    assert response.status_code == 422


def test_search_max_length_rejected(client):
    """search 길이 초과 (40+ 자) → 422."""
    response = client.get(
        "/api/v1/stocks",
        params={"search": "A" * 100},
        headers=FIXED_HEADERS,
    )
    assert response.status_code == 422


def test_search_special_chars_rejected(client):
    """search 특수문자 (SQL fragment 등) → 422."""
    response = client.get(
        "/api/v1/stocks",
        params={"search": "' OR 1=1--"},
        headers=FIXED_HEADERS,
    )
    assert response.status_code == 422


def test_timeframe_invalid_rejected(client):
    """timeframe Literal 외 값 (예: 1M) → 422."""
    response = client.get(
        "/api/v1/stocks/AAPL/prices",
        params={"timeframe": "1M"},
        headers=FIXED_HEADERS,
    )
    assert response.status_code == 422


def test_validation_error_response_schema(client):
    """검증 실패 응답이 통일 schema (error.code, error.message, meta.request_id)
    유지 + details 가 minimal (loc/type 만, raw pydantic 메시지 노출 X)."""
    response = client.get(
        "/api/v1/predictions/line/lowercase",
        headers=FIXED_HEADERS,
    )
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "meta" in body
    # details 가 minimal — ctx, msg, input 등 raw 필드 노출 안 됨
    details = body["error"].get("details", [])
    if details:
        for d in details:
            # 우리가 박은 키만 존재해야 (loc, type). ctx, msg, input 노출 안 됨.
            assert set(d.keys()).issubset({"loc", "type"}), f"details leak: {d.keys()}"
