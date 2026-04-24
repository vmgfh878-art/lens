from __future__ import annotations

from datetime import date, datetime, timedelta

from backend.collector.repositories.base import get_latest_date, upsert_frame
from backend.collector.repositories.sync_state_repo import upsert_job_state
from backend.collector.sources.macro import build_macro_frame
from backend.collector.utils.logging import log


def resolve_macro_start_date(
    lookback_days: int,
    latest_date: date | None,
    *,
    repair_mode: bool = False,
    full_start_date: str | None = None,
    overlap_days: int = 30,
) -> str:
    """거시 데이터 조회 시작일을 결정한다."""
    if repair_mode:
        return full_start_date or (datetime.now().date() - timedelta(days=lookback_days)).isoformat()

    if latest_date:
        return (latest_date - timedelta(days=overlap_days)).isoformat()

    return full_start_date or (datetime.now().date() - timedelta(days=lookback_days)).isoformat()


def run(
    lookback_days: int,
    fred_api_key: str | None,
    fmp_api_key: str | None = None,
    *,
    repair_mode: bool = False,
    full_start_date: str | None = None,
    overlap_days: int = 30,
) -> dict:
    """거시 데이터를 조회해 적재한다."""
    latest_date = get_latest_date("macroeconomic_indicators")
    start_date = resolve_macro_start_date(
        lookback_days,
        latest_date,
        repair_mode=repair_mode,
        full_start_date=full_start_date,
        overlap_days=overlap_days,
    )

    log(
        "거시 데이터 동기화 시작",
        event="sync_macro_started",
        job="sync_macro",
        start_date=start_date,
        repair_mode=repair_mode,
    )
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
        meta={"start_date": start_date, "repair_mode": repair_mode},
    )
    return {"stored": len(frame), "start_date": start_date, "repair_mode": repair_mode}
