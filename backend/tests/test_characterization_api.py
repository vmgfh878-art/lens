"""CP223 백엔드 read-path characterization snapshot 테스트.

9개 v1 endpoint의 정상 200 응답을 CP237.5 drift-resilient 정규화 거쳐
syrupy 로 박제. 코드 변경에 의한 schema/keys/dtypes/list len/row_schema
변동은 잡되, daily refresh 의 새 row 추가 / scalar 값 변동에는 면역.

CP237.5 정규화 (전략 C, `_snapshot_normalize.normalize_response`):
- top-level: status_code + sorted keys + per-key dtype
- list: {len, row_schema} 만 (row 값 안 박음)
- dict: {keys, dtypes} 만 (depth 제한 1)
- scalar: dtype 만 (value 안 박음)

비결정성 회피 (CP223 그대로):
- `X-Request-Id="test-fixed"` 헤더 (request_id.py:9의 uuid4 박제).
- `aapl_prices` 의 start/end 는 테스트 실행 시점 기준 상대 날짜(최근 180일)로 계산.
  CP237.5 정규화가 scalar 값(날짜 포함)을 안 박고 dtype만 비교하므로 절대 날짜를
  고정할 필요가 없다 — 오히려 절대 날짜 고정은 `backend/data/v1/*.parquet`가
  롤링 윈도우(최근 ~13개월만 유지)라 daily refresh가 계속 지나가면 그 구간이
  서빙 데이터 범위 밖으로 밀려나 404가 나는 시한폭탄이었다(2026-08-12 실제 발생,
  api_service.py:58의 date.today() 비결정성 회피 목적으로 2025-01-02~06-30을
  박아뒀던 게 원인). 상대 날짜는 이 클래스의 문제를 구조적으로 없앤다.
- float 은 `normalize_floats(r.json(), ndigits=9)` 로 정규화. 현 정규화 전략에선
  값 비교가 빠져 무해하지만, request_id leak 점검과 향후 fixture 기반 row-level
  비교 회복 시 활용 위해 유지.

운영 코드 무변경. local parquet only (LENS_FORCE_LOCAL=1).

baseline 생성: `pytest backend/tests/test_characterization_api.py --snapshot-update -q`
재실행 (diff=0 검증): `pytest backend/tests/test_characterization_api.py -q`
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from tests._snapshot_normalize import normalize_response
from tests.conftest import FIXED_HEADERS, normalize_floats

_AAPL_PRICES_END = datetime.now().date()
_AAPL_PRICES_START = _AAPL_PRICES_END - timedelta(days=180)

# 9개 케이스. (id, method, path, params).
# AAPL은 9개 모두 200을 반환함을 directive 진단 단계에서 실측 확인.
CASES = [
    ("stocks_list", "/api/v1/stocks", {"limit": 50}),
    (
        "aapl_prices",
        "/api/v1/stocks/AAPL/prices",
        # 최근 180일 상대 윈도 — 서빙 데이터의 롤링 윈도(최근 ~13개월) 안에 항상 들어옴.
        {"start": _AAPL_PRICES_START.isoformat(), "end": _AAPL_PRICES_END.isoformat()},
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
    """9개 endpoint 응답 schema 박제. CP237.5 drift-resilient 정규화 적용."""
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

    normalized = normalize_response(resp.status_code, payload)
    assert normalized == snapshot
