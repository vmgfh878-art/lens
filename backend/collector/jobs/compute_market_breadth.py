from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from backend.collector.repositories.base import fetch_frame, upsert_frame
from backend.collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from backend.collector.utils.logging import log


BREADTH_SOURCE_HISTORY_DAYS = 400
BREADTH_OUTPUT_OVERLAP_DAYS = 14
BREADTH_REPAIR_LOOKBACK_DAYS = 3650
BREADTH_REPAIR_WINDOW_DAYS = 365


def _calculate_stats(frame: pd.DataFrame, min_ticker_count: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"])

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values(["ticker", "date"])
    working["high_52w"] = working.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=252, min_periods=200).max()
    )
    working["low_52w"] = working.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=252, min_periods=200).min()
    )
    working["ma_200"] = working.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=200, min_periods=150).mean()
    )
    working = working.dropna(subset=["high_52w", "low_52w", "ma_200"])
    if working.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"])

    working["is_nh"] = working["close"] >= working["high_52w"]
    working["is_nl"] = working["close"] <= working["low_52w"]
    working["is_above_ma200"] = working["close"] > working["ma_200"]

    stats = (
        working.groupby("date")
        .agg(
            total_count=("ticker", "count"),
            nh_count=("is_nh", "sum"),
            nl_count=("is_nl", "sum"),
            above_ma200_count=("is_above_ma200", "sum"),
        )
        .reset_index()
    )
    stats["nh_nl_index"] = stats["nh_count"] - stats["nl_count"]
    stats["ma200_pct"] = (stats["above_ma200_count"] / stats["total_count"]) * 100.0
    stats = stats[stats["total_count"] >= min_ticker_count][["date", "nh_nl_index", "ma200_pct"]]
    stats["date"] = pd.to_datetime(stats["date"])
    return stats


def _iter_repair_windows(start_date: date, end_date: date, window_days: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + timedelta(days=window_days - 1), end_date)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _run_repair_mode(min_ticker_count: int, window_days: int) -> dict:
    repair_start = datetime.now().date() - timedelta(days=BREADTH_REPAIR_LOOKBACK_DAYS)
    repair_end = datetime.now().date()
    total_rows = 0
    processed_windows = 0
    latest_cursor = None

    for window_start, window_end in _iter_repair_windows(repair_start, repair_end, window_days):
        query_start = window_start - timedelta(days=BREADTH_SOURCE_HISTORY_DAYS)
        price_frame = fetch_frame(
            "price_data",
            columns="date,ticker,close",
            filters=[("gte", "date", query_start.isoformat()), ("lte", "date", window_end.isoformat())],
            order_by="date",
        )
        stats = _calculate_stats(price_frame, min_ticker_count=min_ticker_count)
        if not stats.empty:
            stats = stats[
                (stats["date"].dt.date >= window_start)
                & (stats["date"].dt.date <= window_end)
            ]
        if stats.empty:
            processed_windows += 1
            continue

        latest_cursor = stats["date"].max().date()
        total_rows += len(stats)
        processed_windows += 1
        upsert_frame("market_breadth", stats, on_conflict="date")
        log(
            "market breadth 구간 백필 완료",
            event="compute_market_breadth_window_finished",
            job="compute_market_breadth",
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            stored=len(stats),
        )

    upsert_job_state(
        job_name="compute_market_breadth",
        status="success",
        target_key="__all__",
        last_cursor_date=latest_cursor,
        message=f"stored={total_rows}",
        meta={
            "repair_mode": True,
            "window_days": window_days,
            "processed_windows": processed_windows,
            "source_start_date": repair_start.isoformat(),
            "source_end_date": repair_end.isoformat(),
            "min_ticker_count": min_ticker_count,
        },
    )
    return {
        "stored": total_rows,
        "repair_mode": True,
        "processed_windows": processed_windows,
        "source_start_date": repair_start.isoformat(),
        "source_end_date": repair_end.isoformat(),
    }


def run(repair_mode: bool = False, min_ticker_count: int = 50, repair_window_days: int = BREADTH_REPAIR_WINDOW_DAYS) -> dict:
    """market breadth를 증분 또는 역사 백필 방식으로 계산한다."""
    if repair_mode:
        log(
            "market breadth 전체 백필 시작",
            event="compute_market_breadth_repair_started",
            job="compute_market_breadth",
            repair_window_days=repair_window_days,
            min_ticker_count=min_ticker_count,
        )
        return _run_repair_mode(min_ticker_count=min_ticker_count, window_days=repair_window_days)

    state_map = get_job_state_map("compute_market_breadth")
    state = state_map.get("__all__", {})
    last_cursor_date = pd.to_datetime(state["last_cursor_date"]).date() if state.get("last_cursor_date") else None

    if last_cursor_date is None:
        source_start_date = (datetime.now().date() - timedelta(days=BREADTH_REPAIR_LOOKBACK_DAYS)).isoformat()
        upsert_cutoff = None
    else:
        source_start_date = (last_cursor_date - timedelta(days=BREADTH_SOURCE_HISTORY_DAYS)).isoformat()
        upsert_cutoff = last_cursor_date - timedelta(days=BREADTH_OUTPUT_OVERLAP_DAYS)

    log(
        "market breadth 계산 시작",
        event="compute_market_breadth_started",
        job="compute_market_breadth",
        source_start_date=source_start_date,
        upsert_cutoff=upsert_cutoff.isoformat() if upsert_cutoff else None,
        min_ticker_count=min_ticker_count,
    )

    price_frame = fetch_frame(
        "price_data",
        columns="date,ticker,close",
        filters=[("gte", "date", source_start_date)],
        order_by="date",
    )
    stats = _calculate_stats(price_frame, min_ticker_count=min_ticker_count)
    if upsert_cutoff is not None and not stats.empty:
        stats = stats[stats["date"].dt.date >= upsert_cutoff]
    if not stats.empty:
        upsert_frame("market_breadth", stats, on_conflict="date")

    latest_cursor = None
    if not stats.empty:
        latest_cursor = stats["date"].max().date()
    upsert_job_state(
        job_name="compute_market_breadth",
        status="success",
        target_key="__all__",
        last_cursor_date=latest_cursor,
        message=f"stored={len(stats)}",
        meta={
            "repair_mode": False,
            "source_start_date": source_start_date,
            "upsert_cutoff": upsert_cutoff.isoformat() if upsert_cutoff else None,
            "min_ticker_count": min_ticker_count,
        },
    )
    return {
        "stored": len(stats),
        "source_start_date": source_start_date,
        "upsert_cutoff": upsert_cutoff.isoformat() if upsert_cutoff else None,
        "repair_mode": False,
    }
