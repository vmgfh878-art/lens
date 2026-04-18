from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from collector.repositories.base import fetch_frame, upsert_frame
from collector.repositories.sync_state_repo import upsert_job_state
from collector.utils.logging import log


def _calculate_stats(frame: pd.DataFrame, min_ticker_count: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"])

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])
    frame["high_52w"] = frame.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=252, min_periods=200).max()
    )
    frame["low_52w"] = frame.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=252, min_periods=200).min()
    )
    frame["ma_200"] = frame.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=200, min_periods=150).mean()
    )
    frame = frame.dropna(subset=["high_52w", "low_52w", "ma_200"])
    if frame.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"])

    frame["is_nh"] = frame["close"] >= frame["high_52w"]
    frame["is_nl"] = frame["close"] <= frame["low_52w"]
    frame["is_above_ma200"] = frame["close"] > frame["ma_200"]

    stats = (
        frame.groupby("date")
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
    stats["date"] = pd.to_datetime(stats["date"]).dt.strftime("%Y-%m-%d")
    return stats


def run(repair_mode: bool = False, min_ticker_count: int = 50) -> dict:
    """price_data를 기반으로 market_breadth를 다시 계산한다."""
    lookback_days = 3650 if repair_mode else 400
    start_date = (datetime.now().date() - timedelta(days=lookback_days)).isoformat()
    log(f"[compute_market_breadth] start_date={start_date}, min_ticker_count={min_ticker_count}")

    price_frame = fetch_frame(
        "price_data",
        columns="date,ticker,close",
        filters=[("gte", "date", start_date)],
        order_by="date",
    )
    stats = _calculate_stats(price_frame, min_ticker_count=min_ticker_count)
    upsert_frame("market_breadth", stats, on_conflict="date")

    latest_cursor = None
    if not stats.empty:
        latest_cursor = date.fromisoformat(stats["date"].max())
    upsert_job_state(
        job_name="compute_market_breadth",
        status="success",
        last_cursor_date=latest_cursor,
        message=f"stored={len(stats)}",
    )
    return {"stored": len(stats), "start_date": start_date}
