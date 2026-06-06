"""CP237.5 — Drift-resilient snapshot 정규화 (재귀).

전략 C (Step 1+6 실측 분석 후 확정):
- 값 자체 (scalar / list row / dict value) 는 비교에서 제외.
- 구조 (dict keys + per-key recursive shape + list 의 first-row recursive shape)
  만 비교.
- `len` 도 안 박음 — cumulative list (backtest_aapl.points 등) 의 daily +1행
  증가에 면역.

배경: 지시서 §2의 last_n=5 + scalar value + 1-level dict 전략은 daily refresh 에
fragile. 실측 후 (Step 1): 응답이 `{data: dict{data: list[row...]}}` 중첩이라
1-level shallow dict 는 inner list 의 row schema 를 못 본다. 재귀로 전체 깊이의
구조 박는 형태로 재작성.

목적: daily refresh (새 row 추가 / scalar 값 변동 / cumulative list 증가) 에 면역.
보안/리팩토링 트랙 코드 변경에 의한 응답 keys/dtypes/row_schema 변동은 그대로 검출.

검출 가능 회귀:
- top-level dict 의 key 추가/제거 / value dtype 변경
- nested dict 의 key 추가/제거 / value dtype 변경
- list 의 row dtype 변경 (scalar list)
- list[dict] 의 첫 row 의 key 추가/제거 / value dtype 변경
- status code 변경

검출 불가 (trade-off — 별도 fixture 테스트로 보강):
- row 의 정밀 float 값 회귀
- list 길이 변경 (cumulative 증가 면역의 대가)
- heterogeneous list (row 마다 dtype 다른 경우; 첫 row 기준)
- 첫 row 의 nullable 필드가 dtype 바뀜 (오늘 str / 내일 None)
"""

from __future__ import annotations

from typing import Any

# Step 1 실측 발견 + 미래 대비 변동 키 기록.
# 전략 C 에선 row 값 자체를 안 박아서 이 frozenset 의 실용 가치가 약함.
# 향후 fixture 기반 row-level 비교 도입 시 활용 위해 기록.
DRIFT_FIELDS: frozenset[str] = frozenset(
    {
        "asof_date",
        "asofDate",
        "actual_return",
        "actual_return_available",
        "actual_h5_return",
        "forecast_date",
        "date",
        "line_rank_by_date",
        "safe_line_rank_by_date",
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
    """간단한 dtype label."""
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


def _shape(value: Any) -> Any:
    """재귀 shape 추출.

    - dict → {"dict_keys": sorted_keys, "dict_schema": {k: _shape(v)}}
    - list → {"list_row_schema": _shape(first_row_or_empty)}
    - scalar → dtype 이름 문자열

    JSON 응답은 finite tree (cycle 없음) → 무한 재귀 위험 없음.
    """
    if isinstance(value, dict):
        keys = sorted(value.keys())
        return {
            "dict_keys": keys,
            "dict_schema": {k: _shape(value[k]) for k in keys},
        }
    if isinstance(value, list):
        # 빈 list 는 빈 dict. 첫 row 기준 (heterogeneous list trade-off).
        return {
            "list_row_schema": _shape(value[0]) if value else {},
        }
    return _dtype_name(value)


def normalize_response(status_code: int, payload: Any) -> dict[str, Any]:
    """HTTP 응답을 drift-resilient snapshot 형태로 변환.

    Args:
        status_code: HTTP status code (정수).
        payload: response.json() 결과 (dict / list / scalar).

    Returns:
        snapshot 비교용 dict: {status_code, shape}.
        shape 은 응답 전체 구조의 재귀 정규화 결과.
    """
    return {
        "status_code": status_code,
        "shape": _shape(payload),
    }
