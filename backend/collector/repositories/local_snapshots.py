from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"


def snapshot_dir() -> Path:
    return Path(os.environ.get("LENS_LOCAL_SNAPSHOT_DIR", str(DEFAULT_SNAPSHOT_DIR)))


def local_snapshots_required() -> bool:
    mode = os.environ.get("LENS_DATA_BACKEND", "").strip().lower()
    if mode in {"local", "parquet", "snapshot"}:
        return True
    return os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS", "").strip().lower() in {"1", "true", "yes", "on"}


def local_snapshots_enabled() -> bool:
    flag = os.environ.get("LENS_USE_LOCAL_SNAPSHOTS", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return local_snapshots_required()


def _parse_columns(columns: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if columns is None or columns == "*":
        return None
    if isinstance(columns, str):
        return [column.strip() for column in columns.split(",") if column.strip()]
    return [str(column).strip() for column in columns if str(column).strip()]


def _candidate_paths(table: str, *, provider: str | None = None, timeframe: str | None = None) -> list[Path]:
    base = snapshot_dir()
    names: list[str] = []
    if provider and timeframe:
        names.append(f"{table}_{provider}_{timeframe}.parquet")
    if table == "price_data" and str(timeframe or "").upper() in {"1W", "1M"}:
        if timeframe:
            names.append(f"{table}_{timeframe}.parquet")
        return [base / name for name in names]
    if provider:
        names.append(f"{table}_{provider}.parquet")
    if timeframe:
        names.append(f"{table}_{timeframe}.parquet")
    names.append(f"{table}.parquet")
    candidates = [base / name for name in names]
    candidates.append(base / table)
    return candidates


def _load_snapshot(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return pd.read_parquet(path)
    return pd.read_parquet(path)


def _apply_filters(frame: pd.DataFrame, filters: list[tuple[str, str, Any]] | None) -> pd.DataFrame:
    if frame.empty or not filters:
        return frame
    filtered = frame.copy()
    for operator, column, value in filters:
        if column not in filtered.columns:
            continue
        series = filtered[column]
        if column == "date":
            series = pd.to_datetime(series, errors="coerce")
            compare_value = pd.to_datetime(value, errors="coerce") if value is not None else value
        else:
            compare_value = value

        if operator == "eq":
            filtered = filtered[series == compare_value]
        elif operator == "gte":
            filtered = filtered[series >= compare_value]
        elif operator == "lte":
            filtered = filtered[series <= compare_value]
        elif operator == "in":
            values = set(value or [])
            if column == "ticker":
                values = {str(item).upper() for item in values}
                filtered = filtered[series.astype(str).str.upper().isin(values)]
            else:
                filtered = filtered[series.isin(values)]
        elif operator == "is":
            if value == "null" or value is None:
                filtered = filtered[series.isna()]
        else:
            raise ValueError(f"지원하지 않는 로컬 snapshot 필터입니다: {operator}")
    return filtered


def read_snapshot_frame(
    table: str,
    *,
    columns: str | list[str] | tuple[str, ...] | None = "*",
    filters: list[tuple[str, str, Any]] | None = None,
    order_by: str | None = None,
    ascending: bool = True,
    limit: int | None = None,
    provider: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame | None:
    if not local_snapshots_enabled():
        return None
    existing_path = next(
        (path for path in _candidate_paths(table, provider=provider, timeframe=timeframe) if path.exists()),
        None,
    )
    if existing_path is None:
        if local_snapshots_required():
            raise FileNotFoundError(f"로컬 parquet snapshot을 찾을 수 없습니다: table={table}, dir={snapshot_dir()}")
        return None

    frame = _load_snapshot(existing_path)
    frame = _apply_filters(frame, filters)
    if order_by and order_by in frame.columns:
        frame = frame.sort_values(order_by, ascending=ascending)
    if limit is not None:
        frame = frame.head(limit)

    selected_columns = _parse_columns(columns)
    if selected_columns is not None:
        for column in selected_columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[selected_columns]
    return frame.reset_index(drop=True)
