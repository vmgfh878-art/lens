from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd

from backend.collector.errors import SourceLimitReachedError
from backend.collector.repositories.base import fetch_frame, get_latest_date, upsert_records
from backend.collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from backend.collector.sources.eodhd_prices import attach_trailing_valuation, fetch_ohlcv
from backend.collector.utils.logging import log


EXTREME_RETURN_THRESHOLD = 0.40
DEFAULT_DAILY_LOOKBACK_DAYS = 7


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


def _validate_price_frame(ticker: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """OHLC 무결성과 급격한 이상치를 검사해 적재 대상을 정제한다."""
    if frame.empty:
        return frame, {
            "invalid_ohlc": 0,
            "invalid_volume": 0,
            "extreme_jump": 0,
            "rejected_dates": [],
        }

    working = frame.copy()
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    invalid_ohlc_mask = (
        working["Low"].isna()
        | working["High"].isna()
        | working["Open"].isna()
        | working["Close"].isna()
        | (working["Low"] > working["High"])
        | (working["Open"] < working["Low"])
        | (working["Open"] > working["High"])
        | (working["Close"] < working["Low"])
        | (working["Close"] > working["High"])
    )

    invalid_volume_mask = working["Volume"].isna() | (working["Volume"] <= 0)

    previous_close = working["Close"].shift(1)
    raw_return = (working["Close"] / previous_close - 1.0).abs()

    if "Adj Close" in working.columns:
        previous_adjusted_close = working["Adj Close"].shift(1)
        adjusted_return = (working["Adj Close"] / previous_adjusted_close - 1.0).abs()
        extreme_jump_mask = previous_close.notna() & (raw_return > EXTREME_RETURN_THRESHOLD) & (
            adjusted_return.isna() | (adjusted_return > EXTREME_RETURN_THRESHOLD)
        )
    else:
        extreme_jump_mask = previous_close.notna() & (raw_return > EXTREME_RETURN_THRESHOLD)

    rejected_mask = invalid_ohlc_mask | invalid_volume_mask | extreme_jump_mask
    rejected_dates = [index.date().isoformat() for index in working.index[rejected_mask][:10]]

    if rejected_dates:
        log(
            f"{ticker} 가격 이상치 감지",
            level="WARNING",
            event="price_validation_rejected",
            ticker=ticker,
            invalid_ohlc=int(invalid_ohlc_mask.sum()),
            invalid_volume=int(invalid_volume_mask.sum()),
            extreme_jump=int(extreme_jump_mask.sum()),
            rejected_dates=rejected_dates,
        )

    return working.loc[~rejected_mask].copy(), {
        "invalid_ohlc": int(invalid_ohlc_mask.sum()),
        "invalid_volume": int(invalid_volume_mask.sum()),
        "extreme_jump": int(extreme_jump_mask.sum()),
        "rejected_dates": rejected_dates,
    }


def _build_price_records(ticker: str, frame: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    has_adjusted_close = "Adj Close" in frame.columns

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
                "adjusted_close": get_value("Adj Close") if has_adjusted_close else get_value("Close", 0),
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
    lookback_days: int = DEFAULT_DAILY_LOOKBACK_DAYS,
    repair_mode: bool = False,
    eodhd_api_key: str | None = None,
    batch_limit: int = 80,
    sleep_seconds: float = 0.3,
    allow_yahoo_fallback: bool = False,
    require_fundamentals: bool = False,
) -> dict:
    """일별 가격 데이터를 배치 단위로 동기화한다."""
    today = datetime.now().date().isoformat()
    stored_rows = 0
    skipped_tickers: list[str] = []
    processed_tickers: list[str] = []
    failed_tickers: dict[str, str] = {}
    validation_issues: dict[str, dict] = {}
    quota_hit = False

    stock_rows = fetch_frame("stock_info", columns="ticker", order_by="ticker")
    available_tickers = set(stock_rows["ticker"].tolist()) if not stock_rows.empty else set()
    fundamentals_rows = fetch_frame("company_fundamentals", columns="ticker", order_by="ticker")
    available_fundamentals = set(fundamentals_rows["ticker"].tolist()) if not fundamentals_rows.empty else set()
    fundamentals_cache: dict[str, pd.DataFrame] = {}

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

    log(
        "가격 동기화 배치 시작",
        event="sync_prices_batch_started",
        job="sync_prices",
        batch_size=len(pending_tickers),
        repair_mode=repair_mode,
        lookback_days=lookback_days,
    )

    for ticker in pending_tickers:
        start_date = _get_ticker_start_date(ticker, default_start, lookback_days, repair_mode)
        if not repair_mode and start_date > today:
            skipped_tickers.append(ticker)
            continue

        try:
            price_frame = fetch_ohlcv(
                ticker,
                start_date,
                eodhd_api_key=eodhd_api_key,
                allow_yahoo_fallback=allow_yahoo_fallback,
            )
        except SourceLimitReachedError:
            quota_hit = True
            break

        if price_frame.empty:
            failed_tickers[ticker] = "source_empty"
            upsert_job_state(
                job_name="sync_prices",
                target_key=ticker,
                status="failed",
                message="source_empty",
            )
            continue

        if ticker in available_fundamentals:
            if ticker not in fundamentals_cache:
                fundamentals_cache[ticker] = _load_fundamentals_frame(ticker)
            fundamentals_frame = fundamentals_cache[ticker]
        else:
            fundamentals_frame = pd.DataFrame()

        price_frame = attach_trailing_valuation(price_frame, fundamentals_frame)
        validated_frame, issue_summary = _validate_price_frame(ticker, price_frame)
        if any(issue_summary.values()):
            validation_issues[ticker] = issue_summary

        if validated_frame.empty:
            failed_tickers[ticker] = "all_rows_rejected_by_quality_gate"
            upsert_job_state(
                job_name="sync_prices",
                target_key=ticker,
                status="failed",
                message="all_rows_rejected_by_quality_gate",
                meta=issue_summary,
            )
            continue

        records = _build_price_records(ticker, validated_frame)
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
            meta={"quality_gate": issue_summary},
        )
        time.sleep(sleep_seconds)

    upsert_job_state(
        job_name="sync_prices_summary",
        status="success",
        message=(
            f"stored_rows={stored_rows}, skipped={len(skipped_tickers)}, "
            f"processed={len(processed_tickers)}, failed={len(failed_tickers)}, quota_hit={quota_hit}"
        ),
        meta={
            "skipped": skipped_tickers[:20],
            "processed": processed_tickers[:20],
            "failed": failed_tickers,
            "validation_issues": validation_issues,
        },
    )
    return {
        "stored_rows": stored_rows,
        "skipped": skipped_tickers,
        "processed": processed_tickers,
        "failed": failed_tickers,
        "validation_issues": validation_issues,
        "quota_hit": quota_hit,
        "pending_batch": len(pending_tickers),
        "eligible_fundamentals": len(available_fundamentals),
    }
