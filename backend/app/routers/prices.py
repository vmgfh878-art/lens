from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from app.db import get_supabase
from app.services.model_svc import normalize_timeframe

router = APIRouter()


def _aggregate_prices(rows: list[dict], timeframe: str) -> list[dict]:
    normalized_timeframe = normalize_timeframe(timeframe)
    if normalized_timeframe == "1D" or not rows:
        return rows

    frame = pd.DataFrame(rows)
    if frame.empty:
        return []

    frame["date"] = pd.to_datetime(frame["date"])
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
    aggregated["date"] = aggregated["date"].dt.strftime("%Y-%m-%d")
    return aggregated.to_dict(orient="records")


@router.get("/{ticker}")
def get_prices(
    ticker: str,
    start: str = Query(default="2020-01-01", description="Start date in YYYY-MM-DD format."),
    end: str | None = Query(default=None, description="Optional end date in YYYY-MM-DD format."),
    timeframe: str = Query(default="1D", description="Price timeframe: 1D, 1W, or 1M."),
):
    try:
        normalized_timeframe = normalize_timeframe(timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    client = get_supabase()

    query = (
        client.table("price_data")
        .select("date, open, high, low, close, volume")
        .eq("ticker", ticker.upper())
        .gte("date", start)
        .order("date")
    )
    if end:
        query = query.lte("date", end)

    result = query.execute()
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data found for ticker '{ticker.upper()}'.")

    return {
        "ticker": ticker.upper(),
        "timeframe": normalized_timeframe,
        "data": _aggregate_prices(rows, normalized_timeframe),
    }


@router.get("/")
def get_tickers():
    client = get_supabase()
    result = client.table("stock_info").select("ticker, sector, industry, market_cap").order("ticker").execute()
    return {"tickers": result.data or []}
