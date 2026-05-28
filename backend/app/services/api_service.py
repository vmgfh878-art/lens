from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.core.exceptions import ResourceNotFoundError
from app.services.local_market_svc import (
    fetch_indicator_rows_local,
    fetch_price_rows_local,
    fetch_stocks_local,
)
from app.services.model_svc import normalize_display_timeframe
from app.services.feature_svc import drop_incomplete_resampled_periods

# Supabase 경로는 v1 에서 비활성. 모든 market/stocks 조회는 local parquet 직접 읽음.
# legacy prediction normalization (get_latest_prediction_data 등) 도 같이 제거됨.
# 새 v1 endpoint (/api/v1/predictions/line, /predictions/band/1d, /predictions/band/1w,
# /stocks/{ticker}/predictions/product-history) 가 모든 prediction 조회를 담당한다.


DEFAULT_PRICE_WINDOW_DAYS = 365


def aggregate_prices(rows: list[dict], timeframe: str) -> list[dict]:
    normalized_timeframe = normalize_display_timeframe(timeframe)
    if normalized_timeframe == "1D" or not rows:
        return rows

    frame = pd.DataFrame(rows)
    if frame.empty:
        return []

    frame["date"] = pd.to_datetime(frame["date"])
    latest_daily_date = frame["date"].max()
    frame = frame.sort_values("date").set_index("date")
    rule = "W-FRI" if normalized_timeframe == "1W" else "ME"
    aggregated = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()
    aggregated = drop_incomplete_resampled_periods(
        aggregated,
        normalized_timeframe,
        latest_daily_date=latest_daily_date,
    )
    aggregated["date"] = aggregated["date"].dt.strftime("%Y-%m-%d")
    return aggregated.to_dict(orient="records")


def resolve_price_window(start: str | None, end: str | None) -> tuple[str, str]:
    resolved_end = pd.to_datetime(end).date() if end else date.today()
    resolved_start = pd.to_datetime(start).date() if start else resolved_end - timedelta(days=DEFAULT_PRICE_WINDOW_DAYS)
    if resolved_start > resolved_end:
        raise ValueError("조회 시작일은 종료일보다 늦을 수 없습니다.")
    return resolved_start.isoformat(), resolved_end.isoformat()


def get_stocks(*, search: str | None = None, limit: int = 50) -> list[dict]:
    return fetch_stocks_local(search=search, limit=limit)


def get_price_response_data(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    timeframe: str = "1D",
    limit: int | None = None,
) -> dict:
    normalized_timeframe = normalize_display_timeframe(timeframe)
    resolved_start, resolved_end = resolve_price_window(start, end)
    rows = fetch_price_rows_local(ticker, start=resolved_start, end=resolved_end)
    if not rows:
        raise ResourceNotFoundError(f"종목 '{ticker.upper()}'의 가격 데이터를 찾을 수 없습니다.")

    aggregated = aggregate_prices(rows, normalized_timeframe)
    if limit is not None and limit > 0:
        aggregated = aggregated[-limit:]

    return {
        "ticker": ticker.upper(),
        "timeframe": normalized_timeframe,
        "start": resolved_start,
        "end": resolved_end,
        "data": aggregated,
    }


def get_indicator_response_data(
    ticker: str,
    *,
    timeframe: str = "1D",
    limit: int = 300,
) -> dict:
    normalized_timeframe = normalize_display_timeframe(timeframe)
    rows = fetch_indicator_rows_local(ticker, timeframe=normalized_timeframe, limit=limit)
    return {
        "ticker": ticker.upper(),
        "timeframe": normalized_timeframe,
        "data": rows,
    }


# prediction normalization / legacy endpoint helper 는 v1 에서 제거됨.
