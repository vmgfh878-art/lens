"""
v1 local-parquet 모드용 market 데이터 서빙.

backend/data/v1/market_*.parquet 를 lazy-load 해서 prices / indicators / stocks 응답 생성.
Supabase 미구성 시 api_service 가 이쪽으로 폴백.
"""
from __future__ import annotations

import math
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "data" / "v1"

_PRICES_1D: pd.DataFrame | None = None
_PRICES_1W: pd.DataFrame | None = None
_INDICATORS_1D: pd.DataFrame | None = None
_STOCK_INFO: pd.DataFrame | None = None
_LOCK = Lock()


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def get_prices_1d() -> pd.DataFrame | None:
    global _PRICES_1D
    with _LOCK:
        if _PRICES_1D is None:
            _PRICES_1D = _load(_BASE / "market_prices_1d.parquet")
    return _PRICES_1D


def get_prices_1w() -> pd.DataFrame | None:
    global _PRICES_1W
    with _LOCK:
        if _PRICES_1W is None:
            _PRICES_1W = _load(_BASE / "market_prices_1w.parquet")
    return _PRICES_1W


def get_indicators_1d() -> pd.DataFrame | None:
    global _INDICATORS_1D
    with _LOCK:
        if _INDICATORS_1D is None:
            _INDICATORS_1D = _load(_BASE / "market_indicators_1d.parquet")
    return _INDICATORS_1D


def get_stock_info() -> pd.DataFrame | None:
    global _STOCK_INFO
    with _LOCK:
        if _STOCK_INFO is None:
            _STOCK_INFO = _load(_BASE / "market_stock_info.parquet")
    return _STOCK_INFO


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        try:
            return _jsonable(v.item())
        except Exception:
            return str(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict("records"):
        out.append({k: _jsonable(v) for k, v in row.items()})
    return out


# -------- 공용 API --------


def fetch_price_rows_local(ticker: str, *, start: str, end: str) -> list[dict]:
    """1D prices 만 지원. 1W 는 aggregate_prices 가 1D 에서 만듦."""
    df = get_prices_1d()
    if df is None:
        return []
    ticker = ticker.upper()
    sub = df[(df["ticker"] == ticker) & (df["date"] >= start) & (df["date"] <= end)]
    if sub.empty:
        return []
    return _records(sub[["date", "open", "high", "low", "close", "volume"]].copy())


def fetch_indicator_rows_local(ticker: str, *, timeframe: str = "1D", limit: int = 300) -> list[dict]:
    df = get_indicators_1d()
    if df is None:
        return []
    ticker = ticker.upper()
    sub = df[df["ticker"] == ticker].sort_values("date").tail(limit)
    if sub.empty:
        return []
    return _records(sub.copy())


def fetch_stocks_local(*, search: str | None = None, limit: int = 50) -> list[dict]:
    df = get_stock_info()
    if df is None:
        return []
    sub = df.copy()
    if search:
        s = str(search).upper()
        sub = sub[sub["ticker"].str.startswith(s)]
    sub = sub.head(limit)
    return _records(sub)
