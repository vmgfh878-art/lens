"""CP237.5 — Drift-resilient snapshot 정규화.

전략 C (Step 1 실측 분석 후 확정):
- top-level: status_code / sorted keys / per-key dtype
- list: {len, row_schema} 만 (row 값 자체는 안 박음)
- dict: {keys, dtypes} 만 (depth 제한 1)
- scalar: dtype 만 (value 안 박음)

배경: 지시서 §2의 last_n=5 + scalar value 박는 전략은 daily refresh의 rolling
window 가 매일 row를 swap 시키는 endpoint (aapl_prices/indicators/line/
band_1d/band_1w/product_history/backtest_aapl/scan_indicator) 에 fragile.
실제 응답 schema 실측 후 row-level value / scalar value 비교 자체를 빼는
방향으로 결정. row-level value 회귀 검출력은 별도 fixture 테스트로 보강.

목적: daily refresh 의 새 row 추가 + scalar 값 변동에도 snapshot 안 깨짐.
보안/리팩토링 트랙 코드 변경에 의한 응답 schema/keys/dtypes/list 길이/
row schema 변동은 그대로 검출.
"""

from __future__ import annotations

from typing import Any

# Step 1 실측: 응답에 존재하는 daily 변동 키 + 미래 확장 대비.
# row 값 자체를 안 박는 전략 C 에서는 이 frozenset 의 실용 가치가
# row_schema dict 의 키 정렬 시 비교 안정성 정도 (현재는 row_schema 가
# 모든 키 포함 → 이 frozenset 은 v2 fixture 기반 row-level 비교 시 활용).
# 기록 목적으로 유지.
DRIFT_FIELDS: frozenset[str] = frozenset(
    {
        # Step 1 실측 발견
        "asof_date",
        "asofDate",
        "actual_return",
        "actual_return_available",
        "actual_h5_return",
        "forecast_date",
        "date",
        "line_rank_by_date",
        "safe_line_rank_by_date",
        # 미래 대비 (현재 응답엔 없지만 endpoint 추가 시 흔한 변동 키)
        "actual_h1_return",
        "actual_h4_return",
        "actual_h20_return",
        "created_at",
        "updated_at",
        "inserted_at",
        "latest_asof_date",
        "latest_close_date",
        "data_freshness_date",
        "as_of",
        "computed_at",
        "generated_at",
    }
)


def _dtype_name(value: Any) -> str:
    """간단한 dtype label. Pydantic 응답이 이미 정규화되어 있어 light-weight."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _row_schema(rows: list[Any]) -> dict[str, Any]:
    """list 의 row schema 추출. dict row 면 sorted {key: dtype}, scalar row 면 {_value_dtype}.

    빈 list 는 빈 dict. 첫 row 기준 (heterogeneous list 는 검출 못 함 — trade-off).
    """
    if not rows:
        return {}
    first = rows[0]
    if isinstance(first, dict):
        return {k: _dtype_name(first[k]) for k in sorted(first.keys())}
    return {"_value_dtype": _dtype_name(first)}


def _normalize_list(rows: list[Any]) -> dict[str, Any]:
    """list → {len, row_schema}. row 값 자체는 안 박음 (drift 면역)."""
    return {"len": len(rows), "row_schema": _row_schema(rows)}


def _normalize_dict_shallow(d: dict[str, Any]) -> dict[str, Any]:
    """dict 1-level: sorted keys + per-key dtype. value 안 박음."""
    return {
        "keys": sorted(d.keys()),
        "dtypes": {k: _dtype_name(d[k]) for k in sorted(d.keys())},
    }


def normalize_response(status_code: int, payload: Any) -> dict[str, Any]:
    """HTTP 응답을 drift-resilient snapshot 형태로 변환.

    Args:
        status_code: HTTP status code (정수)
        payload: response.json() 결과 (dict 또는 list)

    Returns:
        snapshot 비교용 dict. scalar value / list row value / nested dict value
        모두 비교 대상에서 제외. schema (keys/dtypes/len/row_schema) 만 비교.
    """
    out: dict[str, Any] = {
        "status_code": status_code,
    }

    if isinstance(payload, dict):
        out["payload_type"] = "dict"
        out["top_level_keys"] = sorted(payload.keys())
        out["top_level_dtypes"] = {k: _dtype_name(payload[k]) for k in sorted(payload.keys())}
        for k in sorted(payload.keys()):
            v = payload[k]
            if isinstance(v, list):
                out[f"list:{k}"] = _normalize_list(v)
            elif isinstance(v, dict):
                out[f"dict:{k}"] = _normalize_dict_shallow(v)
            else:
                # scalar: value 안 박음. dtype 만.
                out[f"scalar:{k}"] = _dtype_name(v)
    elif isinstance(payload, list):
        out["payload_type"] = "list"
        out["root"] = _normalize_list(payload)
    else:
        out["payload_type"] = _dtype_name(payload)
        # 최상위가 scalar 인 응답은 거의 없지만 안전망. value 안 박음.

    return out
