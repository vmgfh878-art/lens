from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from backend.collector.repositories.base import fetch_all_rows, fetch_frame, upsert_records

PRODUCT_LATEST_ALLOWED_LAYERS = {"line", "band"}
PRODUCT_LATEST_DEFAULT_MAX_ROWS = 100
STORAGE_CONTRACT_PRODUCT_LATEST_ONLY = "product_latest_only"
STORAGE_CONTRACT_EVALUATION_BULK = "evaluation_bulk"
PRODUCT_LATEST_TIE_POLICY = "max_asof_date_then_max_decision_time_then_lexical_run_model"


def save_model_run(record: dict[str, Any]) -> None:
    upsert_records("model_runs", [record], on_conflict="run_id")


def get_model_run(run_id: str) -> dict[str, Any] | None:
    rows = fetch_all_rows("model_runs", filters=[("eq", "run_id", run_id)], limit=1)
    return rows[0] if rows else None


def save_predictions(records: list[dict[str, Any]]) -> None:
    upsert_records(
        "predictions",
        records,
        on_conflict="run_id,ticker,model_name,timeframe,horizon,asof_date",
    )


def fetch_run_predictions(run_id: str, timeframe: str | None = None) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("eq", "run_id", run_id)]
    if timeframe:
        filters.append(("eq", "timeframe", timeframe))
    return fetch_frame("predictions", filters=filters, order_by="asof_date")


def save_prediction_evaluations(records: list[dict[str, Any]]) -> None:
    upsert_records(
        "prediction_evaluations",
        records,
        on_conflict="run_id,ticker,timeframe,asof_date",
    )


def _meta_value(record: dict[str, Any], key: str) -> Any:
    meta = record.get("meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def _series_length(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    raise ValueError(f"{key}는 배열 또는 빈 값이어야 합니다.")


def _require_non_empty_series(record: dict[str, Any], key: str, *, layer: str) -> None:
    if _series_length(record, key) <= 0:
        raise ValueError(f"{layer} prediction은 {key}가 1개 이상 있어야 합니다.")


def _require_empty_series(record: dict[str, Any], key: str, *, layer: str) -> None:
    if _series_length(record, key) > 0:
        raise ValueError(f"{layer} prediction에는 {key}를 저장할 수 없습니다.")


def _validate_product_latest_layer_payload(record: dict[str, Any], layer: str) -> None:
    if layer == "line":
        _require_non_empty_series(record, "line_series", layer=layer)
        _require_empty_series(record, "lower_band_series", layer=layer)
        _require_empty_series(record, "upper_band_series", layer=layer)
        return
    if layer == "band":
        _require_non_empty_series(record, "lower_band_series", layer=layer)
        _require_non_empty_series(record, "upper_band_series", layer=layer)
        _require_empty_series(record, "line_series", layer=layer)
        _require_empty_series(record, "conservative_series", layer=layer)
        return
    raise ValueError(f"제품 latest-only prediction meta.layer는 line/band만 허용합니다: {layer}")


def with_prediction_storage_contract(records: list[dict[str, Any]], storage_contract: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        meta = copied.get("meta")
        copied["meta"] = {**(meta if isinstance(meta, dict) else {}), "storage_contract": storage_contract}
        annotated.append(copied)
    return annotated


def _product_latest_layer(record: dict[str, Any]) -> str:
    layer = _meta_value(record, "layer")
    if layer not in PRODUCT_LATEST_ALLOWED_LAYERS:
        raise ValueError(f"제품 latest-only prediction meta.layer는 line/band만 허용합니다: {layer}")
    return str(layer)


def _sortable_text(value: Any) -> str:
    return "" if value is None else str(value)


def _product_latest_group_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("ticker") or "").upper(),
        str(record.get("timeframe") or ""),
        str(record.get("horizon") or ""),
        _product_latest_layer(record),
    )


def _product_latest_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _sortable_text(record.get("asof_date")),
        _sortable_text(record.get("decision_time")),
        _sortable_text(record.get("run_id")),
        _sortable_text(record.get("model_name")),
    )


def select_product_latest_payload(
    prediction_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """제품 latest-only 저장 전 ticker/timeframe/horizon/layer별 최신 row를 고른다."""
    latest_predictions: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in prediction_records:
        key = _product_latest_group_key(record)
        previous = latest_predictions.get(key)
        if previous is None or _product_latest_sort_key(record) > _product_latest_sort_key(previous):
            latest_predictions[key] = record

    selected_predictions = sorted(
        latest_predictions.values(),
        key=lambda item: (_product_latest_group_key(item), _product_latest_sort_key(item)),
    )
    selected_eval_keys = {
        (
            str(record.get("run_id") or ""),
            str(record.get("ticker") or "").upper(),
            str(record.get("timeframe") or ""),
            str(record.get("asof_date") or ""),
        )
        for record in selected_predictions
    }
    latest_evaluations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in evaluation_records:
        key = (
            str(record.get("run_id") or ""),
            str(record.get("ticker") or "").upper(),
            str(record.get("timeframe") or ""),
            str(record.get("asof_date") or ""),
        )
        if key not in selected_eval_keys:
            continue
        previous = latest_evaluations.get(key)
        if previous is None or _sortable_text(record.get("created_at")) > _sortable_text(previous.get("created_at")):
            latest_evaluations[key] = record

    selected_evaluations = sorted(
        latest_evaluations.values(),
        key=lambda item: (
            str(item.get("ticker") or "").upper(),
            str(item.get("timeframe") or ""),
            str(item.get("asof_date") or ""),
        ),
    )
    audit = {
        "contract": STORAGE_CONTRACT_PRODUCT_LATEST_ONLY,
        "selection_scope": "ticker_timeframe_horizon_layer_latest",
        "tie_policy": PRODUCT_LATEST_TIE_POLICY,
        "input_prediction_row_count": len(prediction_records),
        "reduced_prediction_row_count": len(selected_predictions),
        "input_evaluation_row_count": len(evaluation_records),
        "reduced_evaluation_row_count": len(selected_evaluations),
        "ticker_count": len({str(record.get("ticker") or "").upper() for record in selected_predictions}),
        "asof_date_count": len({str(record.get("asof_date") or "") for record in selected_predictions}),
        "layers": sorted({_product_latest_layer(record) for record in selected_predictions}) if selected_predictions else [],
    }
    return selected_predictions, selected_evaluations, audit


def _validate_product_latest_predictions(records: list[dict[str, Any]], *, max_rows: int) -> None:
    if len(records) > max_rows:
        raise ValueError(f"제품 latest-only prediction 저장 row 수가 제한을 초과했습니다: {len(records)} > {max_rows}")
    if not records:
        return

    asof_dates = {str(record.get("asof_date")) for record in records}
    if len(asof_dates) != 1:
        raise ValueError(f"제품 latest-only 저장은 단일 asof_date만 허용합니다: {sorted(asof_dates)}")

    duplicate_keys: set[tuple[Any, ...]] = set()
    for record in records:
        layer = _meta_value(record, "layer")
        composite = bool(_meta_value(record, "composite"))
        model_name = str(record.get("model_name") or "")
        if composite or model_name == "line_band_composite" or layer == "composite":
            raise ValueError("제품 latest-only 저장에서는 composite prediction 저장을 허용하지 않습니다.")
        if layer not in PRODUCT_LATEST_ALLOWED_LAYERS:
            raise ValueError(f"제품 latest-only prediction meta.layer는 line/band만 허용합니다: {layer}")
        _validate_product_latest_layer_payload(record, str(layer))
        key = (
            record.get("run_id"),
            record.get("ticker"),
            record.get("model_name"),
            record.get("timeframe"),
            record.get("horizon"),
            record.get("asof_date"),
        )
        if key in duplicate_keys:
            raise ValueError(f"제품 latest-only prediction 입력에 중복 key가 있습니다: {key}")
        duplicate_keys.add(key)


def _validate_product_latest_evaluations(records: list[dict[str, Any]], *, max_rows: int) -> None:
    if len(records) > max_rows:
        raise ValueError(f"제품 latest-only evaluation 저장 row 수가 제한을 초과했습니다: {len(records)} > {max_rows}")
    if not records:
        return
    asof_dates = {str(record.get("asof_date")) for record in records}
    if len(asof_dates) != 1:
        raise ValueError(f"제품 latest-only evaluation 저장은 단일 asof_date만 허용합니다: {sorted(asof_dates)}")


def save_product_latest_predictions(
    prediction_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
    *,
    max_prediction_rows: int = PRODUCT_LATEST_DEFAULT_MAX_ROWS,
    max_evaluation_rows: int = PRODUCT_LATEST_DEFAULT_MAX_ROWS,
) -> None:
    """제품 화면용 최신 prediction만 얇게 저장한다.

    일반 inference --save와 달리 full history 저장, composite 저장, 과도한 row 저장을 차단한다.
    """
    _validate_product_latest_predictions(prediction_records, max_rows=max_prediction_rows)
    _validate_product_latest_evaluations(evaluation_records, max_rows=max_evaluation_rows)
    save_predictions(with_prediction_storage_contract(prediction_records, STORAGE_CONTRACT_PRODUCT_LATEST_ONLY))
    save_prediction_evaluations(evaluation_records)


def _validate_product_latest_predictions(records: list[dict[str, Any]], *, max_rows: int) -> None:
    if len(records) > max_rows:
        raise ValueError(f"제품 latest-only prediction row 수가 제한을 초과했습니다: {len(records)} > {max_rows}")
    duplicate_keys: set[tuple[Any, ...]] = set()
    for record in records:
        layer = _product_latest_layer(record)
        composite = bool(_meta_value(record, "composite"))
        model_name = str(record.get("model_name") or "")
        if composite or model_name == "line_band_composite" or layer == "composite":
            raise ValueError("제품 latest-only 저장에는 composite prediction을 허용하지 않습니다.")
        _validate_product_latest_layer_payload(record, layer)
        key = (
            str(record.get("ticker") or "").upper(),
            str(record.get("timeframe") or ""),
            str(record.get("horizon") or ""),
            str(record.get("asof_date") or ""),
            layer,
        )
        if key in duplicate_keys:
            raise ValueError(f"제품 latest-only prediction 입력에 중복 key가 있습니다: {key}")
        duplicate_keys.add(key)


def _validate_product_latest_evaluations(records: list[dict[str, Any]], *, max_rows: int) -> None:
    if len(records) > max_rows:
        raise ValueError(f"제품 latest-only evaluation row 수가 제한을 초과했습니다: {len(records)} > {max_rows}")


def save_product_latest_predictions(
    prediction_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
    *,
    max_prediction_rows: int = PRODUCT_LATEST_DEFAULT_MAX_ROWS,
    max_evaluation_rows: int = PRODUCT_LATEST_DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """제품 latest-only 저장 계약: ticker/timeframe/horizon/layer별 최신 row만 저장한다."""
    selected_predictions, selected_evaluations, audit = select_product_latest_payload(
        prediction_records,
        evaluation_records,
    )
    _validate_product_latest_predictions(selected_predictions, max_rows=max_prediction_rows)
    _validate_product_latest_evaluations(selected_evaluations, max_rows=max_evaluation_rows)
    save_predictions(with_prediction_storage_contract(selected_predictions, STORAGE_CONTRACT_PRODUCT_LATEST_ONLY))
    save_prediction_evaluations(selected_evaluations)
    return audit


def fetch_run_evaluations(run_id: str, timeframe: str | None = None) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("eq", "run_id", run_id)]
    if timeframe:
        filters.append(("eq", "timeframe", timeframe))
    return fetch_frame("prediction_evaluations", filters=filters, order_by="asof_date")


def save_backtest_results(records: list[dict[str, Any]]) -> None:
    upsert_records(
        "backtest_results",
        records,
        on_conflict="run_id,strategy_name,timeframe",
    )


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
