from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from supabase import Client

from backend.app.db import get_supabase
from db.load_initial_data import chunked_upsert


def get_client() -> Client:
    return get_supabase()


def _apply_filters(query, filters: list[tuple[str, str, Any]] | None):
    if not filters:
        return query

    for operator, column, value in filters:
        if operator == "eq":
            query = query.eq(column, value)
        elif operator == "gte":
            query = query.gte(column, value)
        elif operator == "lte":
            query = query.lte(column, value)
        elif operator == "in":
            query = query.in_(column, value)
        else:
            raise ValueError(f"지원하지 않는 필터입니다: {operator}")
    return query


def fetch_all_rows(
    table: str,
    columns: str = "*",
    filters: list[tuple[str, str, Any]] | None = None,
    order_by: str | None = None,
    ascending: bool = True,
    page_size: int = 1000,
    limit: int | None = None,
) -> list[dict]:
    """Supabase REST를 페이지 단위로 순회하여 전체 행을 읽는다."""
    client = get_client()
    start = 0
    rows: list[dict] = []

    while True:
        query = client.table(table).select(columns)
        query = _apply_filters(query, filters)
        if order_by:
            query = query.order(order_by, desc=not ascending)
        batch_end = start + page_size - 1
        if limit is not None:
            batch_end = min(batch_end, limit - 1)
        query = query.range(start, batch_end)
        result = query.execute()
        batch = result.data or []
        rows.extend(batch)

        if len(batch) < page_size:
            break
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
        start += page_size

    return rows


def fetch_frame(
    table: str,
    columns: str = "*",
    filters: list[tuple[str, str, Any]] | None = None,
    order_by: str | None = None,
    ascending: bool = True,
    page_size: int = 1000,
    limit: int | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        fetch_all_rows(
            table=table,
            columns=columns,
            filters=filters,
            order_by=order_by,
            ascending=ascending,
            page_size=page_size,
            limit=limit,
        )
    )


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    normalized = frame.copy().astype(object)
    if "ticker" in normalized.columns:
        normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    return normalized.where(pd.notnull(normalized), None)


def upsert_records(table: str, records: list[dict], on_conflict: str) -> None:
    if not records:
        return
    chunked_upsert(get_client(), table, records, on_conflict=on_conflict)


def upsert_frame(table: str, frame: pd.DataFrame, on_conflict: str) -> None:
    if frame.empty:
        return
    upsert_records(table, normalize_frame(frame).to_dict(orient="records"), on_conflict=on_conflict)


def get_latest_date(
    table: str,
    date_column: str = "date",
    filters: list[tuple[str, str, Any]] | None = None,
) -> date | None:
    client = get_client()
    query = client.table(table).select(date_column).order(date_column, desc=True).limit(1)
    query = _apply_filters(query, filters)
    result = query.execute()
    rows = result.data or []
    if not rows or not rows[0].get(date_column):
        return None
    return pd.to_datetime(rows[0][date_column]).date()


def list_known_tickers() -> list[str]:
    rows = fetch_all_rows("stock_info", columns="ticker", order_by="ticker")
    return [row["ticker"] for row in rows if row.get("ticker")]
