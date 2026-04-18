from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from backend.app.services.feature_svc import build_features
from collector.repositories.base import fetch_frame, upsert_records
from collector.repositories.sync_state_repo import upsert_job_state
from collector.utils.logging import log


def _build_filters(start_date: str, tickers: list[str] | None = None) -> list[tuple[str, str, object]]:
    filters: list[tuple[str, str, object]] = [("gte", "date", start_date)]
    if tickers:
        filters.append(("in", "ticker", tickers))
    return filters


def run(lookback_days: int, tickers: list[str] | None = None) -> dict:
    """Lens 피처를 다시 계산한다."""
    indicators_frame = fetch_frame("indicators", columns="date", order_by="date", ascending=False, limit=1)
    latest_indicator_date = None
    if not indicators_frame.empty:
        latest_indicator_date = pd.to_datetime(indicators_frame.iloc[0]["date"]).date()

    if latest_indicator_date is None:
        price_latest = fetch_frame("price_data", columns="date", order_by="date", ascending=False, limit=1)
        if not price_latest.empty:
            latest_indicator_date = pd.to_datetime(price_latest.iloc[0]["date"]).date()

    if latest_indicator_date is None or lookback_days <= 0:
        start_date = "2015-01-01"
    else:
        start_date = (latest_indicator_date - timedelta(days=lookback_days)).isoformat()

    log(f"[compute_indicators] start_date={start_date}")
    price_frame = fetch_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume,amount,per,pbr",
        filters=_build_filters(start_date, tickers),
        order_by="date",
    )
    macro_frame = fetch_frame(
        "macroeconomic_indicators",
        columns="date,us10y,yield_spread,vix_close,credit_spread_hy",
        filters=[("gte", "date", start_date)],
        order_by="date",
    )
    breadth_frame = fetch_frame(
        "market_breadth",
        columns="date,nh_nl_index,ma200_pct",
        filters=[("gte", "date", start_date)],
        order_by="date",
    )

    if price_frame.empty:
        upsert_job_state(
            job_name="compute_indicators",
            status="success",
            message="price_data가 없어 계산을 건너뜀",
        )
        return {"stored": 0, "start_date": start_date}

    price_frame["date"] = pd.to_datetime(price_frame["date"])
    if not macro_frame.empty:
        macro_frame["date"] = pd.to_datetime(macro_frame["date"])
    if not breadth_frame.empty:
        breadth_frame["date"] = pd.to_datetime(breadth_frame["date"])

    total_records = 0
    for timeframe in ("1D", "1W", "1M"):
        features = build_features(
            price_df=price_frame,
            macro_df=macro_frame,
            breadth_df=breadth_frame,
            timeframe=timeframe,
        )
        if features.empty:
            continue

        frame = features.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        records = frame.where(pd.notnull(frame), None).to_dict(orient="records")
        upsert_records("indicators", records, on_conflict="ticker,timeframe,date")
        total_records += len(records)

    upsert_job_state(
        job_name="compute_indicators",
        status="success",
        message=f"stored={total_records}",
    )
    return {"stored": total_records, "start_date": start_date}
