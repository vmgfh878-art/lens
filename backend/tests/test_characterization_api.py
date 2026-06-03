"""CP223 백엔드 read-path characterization snapshot 테스트.

9개 v1 endpoint의 정상 200 응답 전체 JSON을 syrupy로 박제. CP225+ 분리
리팩토링 시 응답이 "1바이트도(허용 tolerance 내에서) 안 바뀌었다"를
기계적으로 증명하는 안전망.

비결정성 회피:
- `X-Request-Id="test-fixed"` 헤더 (request_id.py:9의 uuid4 박제).
- `aapl_prices`의 start/end 명시 (api_service.py:58의 date.today() 회피).
- float은 `normalize_floats(r.json(), ndigits=9)`로 정규화 (rtol≈1e-9).

운영 코드 무변경. local parquet only (LENS_FORCE_LOCAL=1).

baseline 생성: `pytest backend/tests/test_characterization_api.py --snapshot-update -q`
재실행 (diff=0 검증): `pytest backend/tests/test_characterization_api.py -q`
"""

from __future__ import annotations

import pytest
from tests.conftest import FIXED_HEADERS, normalize_floats

# 9개 케이스. (id, method, path, params).
# AAPL은 9개 모두 200을 반환함을 directive 진단 단계에서 실측 확인.
CASES = [
    ("stocks_list", "/api/v1/stocks", {"limit": 50}),
    (
        "aapl_prices",
        "/api/v1/stocks/AAPL/prices",
        # start/end 명시 고정 → date.today() 비결정성 회피.
        {"start": "2025-01-02", "end": "2025-06-30"},
    ),
    ("aapl_indicators", "/api/v1/stocks/AAPL/indicators", {"limit": 300}),
    (
        "aapl_product_history",
        "/api/v1/stocks/AAPL/predictions/product-history",
        {"lookback_days": 60},
    ),
    ("aapl_line", "/api/v1/predictions/line/AAPL", {"days": 365}),
    ("aapl_band_1d", "/api/v1/predictions/band/1d/AAPL", {"days": 365}),
    ("aapl_band_1w", "/api/v1/predictions/band/1w/AAPL", {"days": 730}),
    ("scan_indicator", "/api/v1/strategies/indicator_balance_v2/scan", {"limit": 500}),
    (
        "backtest_aapl",
        "/api/v1/strategies/indicator_balance_v2/backtest/AAPL",
        {},
    ),
]


@pytest.mark.parametrize(
    "case_id,path,params",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_endpoint_snapshot(client, snapshot, case_id, path, params):
    """9개 endpoint 응답 박제. 분리 리팩토링 동작 보존 안전망."""
    resp = client.get(path, params=params, headers=FIXED_HEADERS)
    assert (
        resp.status_code == 200
    ), f"[{case_id}] expected 200, got {resp.status_code}: {resp.text[:200]}"

    payload = normalize_floats(resp.json(), ndigits=9)

    # meta.request_id는 FIXED_HEADERS로 "test-fixed" 박제. 추가 점검:
    if isinstance(payload, dict) and "meta" in payload:
        meta = payload["meta"]
        if isinstance(meta, dict) and "request_id" in meta:
            assert (
                meta["request_id"] == "test-fixed"
            ), f"[{case_id}] request_id leak: {meta['request_id']!r}"

    assert payload == snapshot
