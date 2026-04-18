from __future__ import annotations

from datetime import datetime, timedelta

from collector.repositories.base import get_latest_date, upsert_frame
from collector.repositories.sync_state_repo import upsert_job_state
from collector.sources.macro import build_macro_frame
from collector.utils.logging import log


def run(lookback_days: int, fred_api_key: str | None, fmp_api_key: str | None = None) -> dict:
    """거시 지표를 동기화한다."""
    latest_date = get_latest_date("macroeconomic_indicators")
    if latest_date:
        start_date = (latest_date - timedelta(days=30)).isoformat()
    else:
        start_date = (datetime.now().date() - timedelta(days=lookback_days)).isoformat()

    log(f"[sync_macro] start_date={start_date}")
    frame = build_macro_frame(start_date, fred_api_key, fmp_api_key)
    upsert_frame("macroeconomic_indicators", frame, on_conflict="date")

    latest_cursor = None
    if not frame.empty:
        latest_cursor = frame["date"].max().date()
    upsert_job_state(
        job_name="sync_macro",
        status="success",
        last_cursor_date=latest_cursor,
        message=f"stored={len(frame)}",
    )
    return {"stored": len(frame), "start_date": start_date}
