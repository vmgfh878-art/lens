from __future__ import annotations

from datetime import date, datetime, timedelta

from collector.repositories.base import fetch_frame, get_latest_date, upsert_frame
from collector.repositories.sync_state_repo import upsert_job_state
from collector.sources.sector_returns import build_sector_returns
from collector.utils.logging import log


def run(default_lookback_days: int) -> dict:
    """price_data와 stock_info를 이용해 섹터 수익률을 계산한다."""
    latest_date = get_latest_date("sector_returns")
    if latest_date is None:
        days_back = default_lookback_days
    else:
        days_back = max((datetime.now().date() - latest_date).days + 10, 40)

    start_date = (datetime.now().date() - timedelta(days=days_back + 5)).isoformat()
    log(f"[sync_sector_returns] days_back={days_back}")

    price_frame = fetch_frame(
        "price_data",
        columns="date,ticker,close",
        filters=[("gte", "date", start_date)],
        order_by="date",
    )
    stock_info_frame = fetch_frame("stock_info", columns="ticker,sector", order_by="ticker")
    frame = build_sector_returns(price_frame=price_frame, stock_info_frame=stock_info_frame)
    upsert_frame("sector_returns", frame, on_conflict="date,sector")

    latest_cursor = None
    if not frame.empty:
        latest_cursor = date.fromisoformat(frame["date"].max())
    upsert_job_state(
        job_name="sync_sector_returns",
        status="success",
        last_cursor_date=latest_cursor,
        message=f"stored={len(frame)}",
    )
    return {"stored": len(frame), "days_back": days_back}
