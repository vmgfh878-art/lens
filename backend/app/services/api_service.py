from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from app.core.exceptions import ResourceNotFoundError
from app.repositories import market_repo
from app.services import data_backend
from app.services.feature_svc import drop_incomplete_resampled_periods
from app.services.local_market_svc import (
    fetch_indicator_rows_local,
    fetch_price_rows_local,
    fetch_stocks_local,
)
from app.services.model_svc import normalize_display_timeframe

# CP254 — market/stocks read 는 data_backend.use_supabase() 토글:
#   로컬 모드(기본): backend/data/v1 parquet (local_market_svc) — 기존 v1 경로 그대로.
#   REST 모드: market_repo (price_data/indicators/stock_info, 컬럼 화이트리스트).
# source 는 SERVING_SOURCE='yfinance' 고정 (서빙 실제 출처, 2026-06-18 정정). collector
# 기본 provider 도 yfinance 라 라벨 자동 일치 — read/import/크론 source 가 한 값으로 통일.
# legacy prediction normalization (get_latest_prediction_data 등) 은 v1 에서 제거됨.
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
    resolved_start = (
        pd.to_datetime(start).date()
        if start
        else resolved_end - timedelta(days=DEFAULT_PRICE_WINDOW_DAYS)
    )
    if resolved_start > resolved_end:
        raise ValueError("조회 시작일은 종료일보다 늦을 수 없습니다.")
    return resolved_start.isoformat(), resolved_end.isoformat()


def get_stocks(*, search: str | None = None, limit: int = 50) -> list[dict]:
    if data_backend.use_supabase():
        return market_repo.fetch_stocks(
            search=search, limit=limit, source=data_backend.SERVING_SOURCE
        )
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
    if data_backend.use_supabase():
        rows = market_repo.fetch_price_rows(
            ticker,
            start=resolved_start,
            end=resolved_end,
            source=data_backend.SERVING_SOURCE,
        )
    else:
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
    if data_backend.use_supabase():
        # 현행 로컬 계약 보존: 지표는 1D 만 존재하고, 1W 요청도 1D 데이터를
        # 반환한다 (fetch_indicator_rows_local 이 timeframe 무관 1D parquet 읽음).
        # REST 도 동일하게 1D 로 조회해야 parity 가 깨지지 않는다.
        rows = market_repo.fetch_indicator_rows(
            ticker, timeframe="1D", limit=limit, source=data_backend.SERVING_SOURCE
        )
    else:
        rows = fetch_indicator_rows_local(ticker, timeframe=normalized_timeframe, limit=limit)
    return {
        "ticker": ticker.upper(),
        "timeframe": normalized_timeframe,
        "data": rows,
    }


# prediction normalization / legacy endpoint helper 는 v1 에서 제거됨.
