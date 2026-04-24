from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from backend.collector.repositories.base import fetch_all_rows, fetch_frame, upsert_records


def save_model_run(record: dict[str, Any]) -> None:
    upsert_records("model_runs", [record], on_conflict="run_id")


def get_model_run(run_id: str) -> dict[str, Any] | None:
    rows = fetch_all_rows("model_runs", filters=[("eq", "run_id", run_id)], limit=1)
    return rows[0] if rows else None


def save_predictions(records: list[dict[str, Any]]) -> None:
    upsert_records(
        "predictions",
        records,
        on_conflict="ticker,model_name,timeframe,horizon,asof_date",
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
