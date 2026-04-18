from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd

from collector.errors import SourceLimitReachedError
from collector.repositories.base import fetch_frame, get_latest_date, upsert_records
from collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from collector.sources.yahoo_prices import attach_trailing_valuation, fetch_ohlcv
from collector.utils.logging import log


def _get_ticker_start_date(ticker: str, default_start: str, lookback_days: int, repair_mode: bool) -> str:
    if repair_mode:
        return default_start

    latest_date = get_latest_date("price_data", filters=[("eq", "ticker", ticker)])
    if latest_date is None:
        return default_start

    buffered_start = latest_date - timedelta(days=lookback_days)
    default_start_date = date.fromisoformat(default_start)
    return max(buffered_start, default_start_date).isoformat()


def _load_fundamentals_frame(ticker: str) -> pd.DataFrame:
    return fetch_frame(
        "company_fundamentals",
        columns="date,equity,shares_issued,eps",
        filters=[("eq", "ticker", ticker)],
        order_by="date",
    )


def _build_price_records(ticker: str, frame: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    has_adj_close = "Adj Close" in frame.columns

    for index, row in frame.iterrows():
        def get_value(column_name: str, default=None):
            value = row.get(column_name, default)
            if hasattr(value, "iloc"):
                value = value.iloc[0]
            if value is None or pd.isna(value):
                return None
            return float(value)

        records.append(
            {
                "date": index.date().isoformat(),
                "ticker": ticker,
                "open": get_value("Open", 0),
                "high": get_value("High", 0),
                "low": get_value("Low", 0),
                "close": get_value("Close", 0),
                "adjusted_close": get_value("Adj Close") if has_adj_close else get_value("Close", 0),
                "volume": int(get_value("Volume", 0) or 0),
                "amount": get_value("Amount", 0),
                "per": get_value("per"),
                "pbr": get_value("pbr"),
            }
        )
    return records


def run(
    tickers: list[str],
    default_start: str,
    lookback_days: int = 45,
    repair_mode: bool = False,
    fmp_api_key: str | None = None,
    batch_limit: int = 80,
    sleep_seconds: float = 0.3,
    allow_yahoo_fallback: bool = True,
    require_fundamentals: bool = False,
) -> dict:
    """일별 가격 데이터를 배치 단위로 동기화한다."""
    today = datetime.now().date().isoformat()
    stored_rows = 0
    skipped_tickers: list[str] = []
    processed_tickers: list[str] = []
    quota_hit = False

    stock_rows = fetch_frame("stock_info", columns="ticker", order_by="ticker")
    available_tickers = set(stock_rows["ticker"].tolist()) if not stock_rows.empty else set()
    fundamentals_rows = fetch_frame("company_fundamentals", columns="ticker", order_by="ticker")
    available_fundamentals = set(fundamentals_rows["ticker"].tolist()) if not fundamentals_rows.empty else set()

    state_map = get_job_state_map("sync_prices")
    if repair_mode:
        pending_tickers = [
            ticker
            for ticker in tickers
            if ticker in available_tickers
            and (not require_fundamentals or ticker in available_fundamentals)
            and state_map.get(ticker, {}).get("status") != "success"
        ][:batch_limit]
    else:
        pending_tickers = [
            ticker
            for ticker in tickers
            if ticker in available_tickers and (not require_fundamentals or ticker in available_fundamentals)
        ][:batch_limit]

    log(f"[sync_prices] 대상 {len(pending_tickers)}개 종목 배치 시작")
    for ticker in pending_tickers:
        start_date = _get_ticker_start_date(ticker, default_start, lookback_days, repair_mode)
        if not repair_mode and start_date > today:
            skipped_tickers.append(ticker)
            continue

        try:
            price_frame = fetch_ohlcv(
                ticker,
                start_date,
                fmp_api_key=fmp_api_key,
                allow_yahoo_fallback=allow_yahoo_fallback,
            )
        except SourceLimitReachedError:
            quota_hit = True
            break

        if price_frame.empty:
            skipped_tickers.append(ticker)
            upsert_job_state(
                job_name="sync_prices",
                target_key=ticker,
                status="failed",
                message="stored=0",
            )
            continue

        fundamentals_frame = _load_fundamentals_frame(ticker)
        price_frame = attach_trailing_valuation(price_frame, fundamentals_frame)
        records = _build_price_records(ticker, price_frame)
        upsert_records("price_data", records, on_conflict="ticker,date")
        stored_rows += len(records)
        processed_tickers.append(ticker)

        last_date = records[-1]["date"] if records else None
        upsert_job_state(
            job_name="sync_prices",
            target_key=ticker,
            status="success",
            last_cursor_date=date.fromisoformat(last_date) if last_date else None,
            message=f"stored={len(records)}",
        )
        time.sleep(sleep_seconds)

    upsert_job_state(
        job_name="sync_prices_summary",
        status="success",
        message=(
            f"stored_rows={stored_rows}, skipped={len(skipped_tickers)}, "
            f"processed={len(processed_tickers)}, quota_hit={quota_hit}"
        ),
        meta={"skipped": skipped_tickers[:20], "processed": processed_tickers[:20]},
    )
    return {
        "stored_rows": stored_rows,
        "skipped": skipped_tickers,
        "processed": processed_tickers,
        "quota_hit": quota_hit,
        "pending_batch": len(pending_tickers),
        "eligible_fundamentals": len(available_fundamentals),
    }
