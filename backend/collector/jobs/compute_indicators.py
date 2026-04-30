from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

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
TIMEFRAME_BATCH_SIZE = {
    "1D": 120,
    "1W": 80,
    "1M": 40,
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

    known_cursors = [
        pd.to_datetime(state_map[ticker]["last_cursor_date"]).date()
        for ticker in target_tickers
        if ticker in state_map and state_map[ticker].get("last_cursor_date")
    ]
    if not known_cursors:
        return full_start_date

    earliest_cursor = min(known_cursors)
    history_days = TIMEFRAME_CONFIG[timeframe]["source_history_days"]
    return (earliest_cursor - timedelta(days=history_days)).isoformat()


def _resolve_upsert_cutoff(timeframe: str, last_cursor_date: date | None, overlap_days: int) -> date | None:
    if last_cursor_date is None:
        return None
    return last_cursor_date - timedelta(days=max(overlap_days, TIMEFRAME_CONFIG[timeframe]["output_overlap_days"]))


def _fetch_context_frame(table: str, columns: str, start_date: str) -> pd.DataFrame:
    return fetch_frame(
        table,
        columns=columns,
        filters=[("gte", "date", start_date)],
        order_by="date",
    )


def _chunked(values: list[str], size: int) -> list[list[str]]:
    if not values:
        return []
    return [values[index : index + size] for index in range(0, len(values), size)]


def run(
    lookback_days: int,
    tickers: list[str] | None = None,
    *,
    force_full_backfill: bool = False,
    full_start_date: str = "2015-01-01",
    timeframes: Iterable[str] | None = None,
) -> dict:
    """지표를 증분 또는 전체 백필 기준으로 계산한다."""

    target_tickers = tickers or []
    summary: dict[str, dict] = {}
    total_records = 0
    resolved_timeframes = tuple(timeframes or ("1D", "1W", "1M"))
    unsupported = [timeframe for timeframe in resolved_timeframes if timeframe not in TIMEFRAME_CONFIG]
    if unsupported:
        raise ValueError(f"지원하지 않는 indicator timeframe 입니다: {unsupported}")

    for timeframe in resolved_timeframes:
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

        if not macro_frame.empty:
            macro_frame["date"] = pd.to_datetime(macro_frame["date"])
        if not breadth_frame.empty:
            breadth_frame["date"] = pd.to_datetime(breadth_frame["date"])

        timeframe_records = 0
        per_ticker_counts: dict[str, int] = {}
        ticker_batches = _chunked(target_tickers, TIMEFRAME_BATCH_SIZE[timeframe]) if target_tickers else [[]]
        if len(ticker_batches) > 1:
            log(
                "지표 계산 배치 분할",
                event="compute_indicators_batched",
                job="compute_indicators",
                timeframe=timeframe,
                batch_count=len(ticker_batches),
                batch_size=TIMEFRAME_BATCH_SIZE[timeframe],
            )

        for ticker_batch in ticker_batches:
            price_frame = fetch_frame(
                "price_data",
                columns="ticker,date,open,high,low,close,adjusted_close,volume,amount,per,pbr",
                filters=_build_filters(source_start_date, ticker_batch or None),
                order_by="date",
            )
            if price_frame.empty:
                continue

            price_frame["date"] = pd.to_datetime(price_frame["date"])
            fundamentals_filters: list[tuple[str, str, object]] = []
            if ticker_batch:
                fundamentals_filters.append(("in", "ticker", ticker_batch))
            fundamentals_frame = fetch_frame(
                "company_fundamentals",
                columns="ticker,date,filing_date,revenue,net_income,total_liabilities,equity,eps",
                filters=fundamentals_filters,
                order_by="date",
            )
            if not fundamentals_frame.empty:
                fundamentals_frame["date"] = pd.to_datetime(fundamentals_frame["date"])
                fundamentals_frame["filing_date"] = pd.to_datetime(fundamentals_frame["filing_date"])

            features = build_features(
                price_df=price_frame,
                macro_df=macro_frame,
                breadth_df=breadth_frame,
                fundamentals_df=fundamentals_frame,
                timeframe=timeframe,
            )
            if features.empty:
                continue

            for ticker, ticker_frame in features.groupby("ticker", sort=True):
                ticker_state = state_map.get(ticker, {})
                last_cursor_date = (
                    pd.to_datetime(ticker_state["last_cursor_date"]).date()
                    if ticker_state.get("last_cursor_date") and not force_full_backfill
                    else None
                )
                upsert_cutoff = _resolve_upsert_cutoff(timeframe, last_cursor_date, lookback_days)

                output_frame = ticker_frame.copy()
                output_frame["date"] = pd.to_datetime(output_frame["date"])
                if upsert_cutoff is not None:
                    output_frame = output_frame[output_frame["date"].dt.date >= upsert_cutoff]
                if output_frame.empty:
                    continue

                # 같은 날짜가 중복되면 Supabase upsert가 실패하므로 마지막 값만 남긴다.
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
                        "timeframe": timeframe,
                        "upsert_cutoff": upsert_cutoff.isoformat() if upsert_cutoff else None,
                        "force_full_backfill": force_full_backfill,
                    },
                )

        summary[timeframe] = {
            "stored": timeframe_records,
            "source_start_date": source_start_date,
            "ticker_count": len(per_ticker_counts),
            "force_full_backfill": force_full_backfill,
        }
        total_records += timeframe_records

    upsert_job_state(
        job_name="compute_indicators",
        status="success",
        last_cursor_date=datetime.now().date(),
        message=f"stored={total_records}",
        meta={"force_full_backfill": force_full_backfill, "timeframes": summary},
    )
    return {"stored": total_records, "timeframes": summary}
