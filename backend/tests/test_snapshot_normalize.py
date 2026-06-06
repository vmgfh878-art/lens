"""CP237.5 — Drift simulation.

`_snapshot_normalize.normalize_response` 가:
- daily refresh (새 row 추가 / scalar 값 변동 / cumulative list 증가) 에 면역
- 코드 회귀 (key 추가/제거 / dtype 변화 / nested schema / status code) 는 검출

이 두 의도를 분리해서 검증. 지시서 §6 3 케이스 + 보강.

빈도가 빈번한 drift 패턴별로 1 케이스, 회귀 검출 패턴별로 1 케이스씩.
모두 normalize_response 만 호출하는 단위 테스트 — TestClient 불필요.
"""

from __future__ import annotations

from tests._snapshot_normalize import normalize_response


# ---------------------------------------------------------------------------
# Drift 면역 — 매일 일어나는 변동에 normalized 동일
# ---------------------------------------------------------------------------


def test_new_row_added_does_not_break():
    """daily refresh: 새 row 1개 list 끝에 추가."""
    base = {
        "data": [
            {"asof_date": "2025-05-30", "value": 1.0, "ticker": "AAPL"},
            {"asof_date": "2025-05-31", "value": 2.0, "ticker": "AAPL"},
        ],
        "meta": {"count": 2},
    }
    drifted = {
        "data": base["data"] + [{"asof_date": "2025-06-01", "value": 3.0, "ticker": "AAPL"}],
        "meta": {"count": 3},
    }
    assert normalize_response(200, base) == normalize_response(200, drifted)


def test_scalar_value_change_does_not_break():
    """backtest_aapl 의 averageHoldingDays 처럼 top-level scalar 가 매일 변동."""
    base = {"averageHoldingDays": 65.66, "trades": [{"date": "2025-06-01"}]}
    drifted = {"averageHoldingDays": 66.66, "trades": [{"date": "2025-06-02"}]}
    assert normalize_response(200, base) == normalize_response(200, drifted)


def test_cumulative_list_grows_does_not_break():
    """backtest_aapl.points 처럼 list 길이가 매일 +1 증가."""
    base = {"points": [{"date": "d1", "v": 1.0}]}
    drifted = {
        "points": [
            {"date": "d1", "v": 1.0},
            {"date": "d2", "v": 2.0},
            {"date": "d3", "v": 3.0},
        ]
    }
    assert normalize_response(200, base) == normalize_response(200, drifted)


def test_nested_scalar_change_does_not_break():
    """nested dict 안의 scalar 값 변화 (scan_indicator.aggregateMetrics 시뮬)."""
    base = {"data": {"agg": {"return": 12.5, "count": 100}}}
    drifted = {"data": {"agg": {"return": -3.7, "count": 200}}}
    assert normalize_response(200, base) == normalize_response(200, drifted)


# ---------------------------------------------------------------------------
# 회귀 검출 — 코드 변경에 의한 schema 변화는 정확히 잡음
# ---------------------------------------------------------------------------


def test_row_key_added_is_detected():
    """row 에 key 추가 → schema 변화 → 검출."""
    base = {"data": [{"a": 1, "b": 2}]}
    drifted = {"data": [{"a": 1, "b": 2, "c": 3}]}
    assert normalize_response(200, base) != normalize_response(200, drifted)


def test_row_key_removed_is_detected():
    """row 에 key 제거 → schema 변화 → 검출."""
    base = {"data": [{"a": 1, "b": 2}]}
    drifted = {"data": [{"a": 1}]}
    assert normalize_response(200, base) != normalize_response(200, drifted)


def test_dtype_change_is_detected():
    """value dtype (int → str) 변화 → 검출."""
    base = {"data": [{"a": 1}]}
    drifted = {"data": [{"a": "1"}]}
    assert normalize_response(200, base) != normalize_response(200, drifted)


def test_nested_schema_change_is_detected():
    """nested dict (depth 2+) key rename → 재귀 정규화로 검출.

    이게 1-level shallow 정규화 시 검출 불가였던 핵심 함정. 재귀로 해결됨.
    """
    base = {"data": {"data": [{"row_key": 1}]}}
    drifted = {"data": {"data": [{"row_key_renamed": 1}]}}
    assert normalize_response(200, base) != normalize_response(200, drifted)


def test_top_level_key_added_is_detected():
    """top-level dict 에 key 추가 → 검출."""
    base = {"data": []}
    drifted = {"data": [], "extra": []}
    assert normalize_response(200, base) != normalize_response(200, drifted)


def test_status_code_change_is_detected():
    """status code 변화 → 검출."""
    a = normalize_response(200, {"ok": True})
    b = normalize_response(500, {"ok": True})
    assert a != b


def test_empty_list_to_populated_is_detected():
    """list 가 빈 상태 → row 채워진 상태 = row_schema 발생 → 검출.

    의도: 응답이 빈 list 였다가 row 가 추가되는 변화는 schema 진화이므로 검출.
    """
    base = {"data": []}
    drifted = {"data": [{"row_key": 1}]}
    assert normalize_response(200, base) != normalize_response(200, drifted)
