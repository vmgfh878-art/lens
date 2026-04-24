from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from backend.app.services.feature_svc import build_features
from backend.collector.repositories.base import fetch_frame, upsert_records
from backend.collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from backend.collector.utils.logging import log


TIMEFRAME_CONFIG = {
    "1D": {"source_history_days": 250, "output_overlap_days": 14},
    "1W": {"source_history_days": 550, "output_overlap_days": 56},
    "1M": {"source_history_days": 2100, "output_overlap_days": 120},
}


def _build_filters(start_date: str, tickers: list[str] | None = None) -> list[tuple[str, str, object]]:
    filters: list[tuple[str, str, object]] = [("gte", "date", start_date)]
    if tickers:
        filters.append(("in", "ticker", tickers))
    return filters


def _resolve_source_start_date(
    timeframe: str,
    tickers: list[str] | None,
    state_map: dict[str, dict],
    *,
    force_full_backfill: bool = False,
    full_start_date: str = "2015-01-01",
) -> str:
    if force_full_backfill:
        return full_start_date

    target_tickers = tickers or []
    if not target_tickers:
        return full_start_date

    if any(ticker not in state_map or not state_map[ticker].get("last_cursor_date") for ticker in target_tickers):
        return full_start_date

    earliest_cursor = min(pd.to_datetime(state_map[ticker]["last_cursor_date"]).date() for ticker in target_tickers)
    history_days = TIMEFRAME_CONFIG[timeframe]["source_history_days"]
    return (earliest_cursor - timedelta(days=history_days)).isoformat()


def _resolve_upsert_cutoff(timeframe: str, last_cursor_date: date | None, overlap_days: int) -> date | None:
    if last_cursor_date is None:
        return None
    return last_cursor_date - timedelta(days=max(overlap_days, TIMEFRAME_CONFIG[timeframe]["output_overlap_days"]))


def _fetch_context_frame(table: str, columns: str, start_date: str) -> pd.DataFrame:
    frame = fetch_frame(
        table,
        columns=columns,
        filters=[("gte", "date", start_date)],
        order_by="date",
    )
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def run(
    lookback_days: int,
    tickers: list[str] | None = None,
    *,
    force_full_backfill: bool = False,
    full_start_date: str = "2015-01-01",
) -> dict:
    """지표를 증분 또는 전체 백필 기준으로 계산한다."""
    overlap_days = max(1, lookback_days)
    target_tickers = tickers or []
    summary: dict[str, dict] = {}
    total_records = 0

    for timeframe in ("1D", "1W", "1M"):
        state_job_name = f"compute_indicators:{timeframe}"
        state_map = get_job_state_map(state_job_name)
        source_start_date = _resolve_source_start_date(
            timeframe,
            target_tickers,
            state_map,
            force_full_backfill=force_full_backfill,
            full_start_date=full_start_date,
        )

        log(
            "지표 계산 시작",
            event="compute_indicators_started",
            job="compute_indicators",
            timeframe=timeframe,
            source_start_date=source_start_date,
            ticker_count=len(target_tickers),
            force_full_backfill=force_full_backfill,
        )

        price_frame = fetch_frame(
            "price_data",
            columns="ticker,date,open,high,low,close,adjusted_close,volume,amount,per,pbr",
            filters=_build_filters(source_start_date, target_tickers),
            order_by="date",
        )
        if price_frame.empty:
            summary[timeframe] = {"stored": 0, "source_start_date": source_start_date, "ticker_count": 0}
            continue

        price_frame["date"] = pd.to_datetime(price_frame["date"])
        macro_frame = _fetch_context_frame(
            "macroeconomic_indicators",
            "date,us10y,yield_spread,vix_close,credit_spread_hy",
            source_start_date,
        )
        breadth_frame = _fetch_context_frame(
            "market_breadth",
            "date,nh_nl_index,ma200_pct",
            source_start_date,
        )
        fundamentals_filters: list[tuple[str, str, object]] = []
        if target_tickers:
            fundamentals_filters.append(("in", "ticker", target_tickers))
        fundamentals_frame = fetch_frame(
            "company_fundamentals",
            columns="ticker,date,filing_date,revenue,net_income,total_liabilities,equity,eps",
            filters=fundamentals_filters or None,
            order_by="filing_date",
        )
        if not fundamentals_frame.empty:
            fundamentals_frame["date"] = pd.to_datetime(fundamentals_frame["date"])
            fundamentals_frame["filing_date"] = pd.to_datetime(fundamentals_frame["filing_date"], errors="coerce")

        features = build_features(
            price_df=price_frame,
            macro_df=macro_frame,
            breadth_df=breadth_frame,
            fundamentals_df=fundamentals_frame,
            timeframe=timeframe,
        )
        if features.empty:
            summary[timeframe] = {"stored": 0, "source_start_date": source_start_date, "ticker_count": 0}
            continue

        timeframe_records = 0
        per_ticker_counts: dict[str, int] = {}
        for ticker, ticker_frame in features.groupby("ticker", sort=True):
            ticker_state = state_map.get(ticker, {})
            last_cursor_date = (
                pd.to_datetime(ticker_state["last_cursor_date"]).date()
                if ticker_state.get("last_cursor_date") and not force_full_backfill
                else None
            )
            upsert_cutoff = _resolve_upsert_cutoff(timeframe, last_cursor_date, overlap_days)

            output_frame = ticker_frame.copy()
            output_frame["date"] = pd.to_datetime(output_frame["date"])
            if upsert_cutoff is not None:
                output_frame = output_frame[output_frame["date"].dt.date >= upsert_cutoff]
            if output_frame.empty:
                continue

            # 동일 키가 한 청크에 중복되면 Supabase upsert가 실패하므로 마지막 값을 남긴다.
            output_frame = (
                output_frame.sort_values(["ticker", "timeframe", "date"])
                .drop_duplicates(subset=["ticker", "timeframe", "date"], keep="last")
            )
            output_frame["date"] = output_frame["date"].dt.strftime("%Y-%m-%d")
            records = output_frame.where(pd.notnull(output_frame), None).to_dict(orient="records")
            upsert_records("indicators", records, on_conflict="ticker,timeframe,date")
            timeframe_records += len(records)
            per_ticker_counts[ticker] = len(records)

            latest_cursor = date.fromisoformat(records[-1]["date"])
            upsert_job_state(
                job_name=state_job_name,
                target_key=ticker,
                status="success",
                last_cursor_date=latest_cursor,
                message=f"stored={len(records)}",
                meta={
                    "source_start_date": source_start_date,
                    "upsert_cutoff": upsert_cutoff.isoformat() if upsert_cutoff else None,
                    "force_full_backfill": force_full_backfill,
                },
            )

        total_records += timeframe_records
        summary[timeframe] = {
            "stored": timeframe_records,
            "source_start_date": source_start_date,
            "ticker_count": len(per_ticker_counts),
            "force_full_backfill": force_full_backfill,
        }

    upsert_job_state(
        job_name="compute_indicators",
        status="success",
        last_cursor_date=datetime.now().date(),
        message=f"stored={total_records}",
        meta=summary,
    )
    return {"stored": total_records, "timeframes": summary}
