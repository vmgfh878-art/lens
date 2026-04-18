from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from collector.errors import SourceLimitReachedError
from collector.sources.yf_common import prepare_yfinance

prepare_yfinance()


def _fetch_fmp_ohlcv(ticker: str, start_date: str, api_key: str) -> pd.DataFrame:
    url = (
        "https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={ticker}&from={start_date}&apikey={api_key.strip()}"
    )
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 429:
            raise SourceLimitReachedError("FMP", response.text.strip())
        response.raise_for_status()
        payload = response.json()
    except SourceLimitReachedError:
        raise
    except Exception:
        return pd.DataFrame()

    if isinstance(payload, dict) and "Error Message" in payload:
        message = str(payload.get("Error Message", ""))
        if "Limit Reach" in message:
            raise SourceLimitReachedError("FMP", message)
        return pd.DataFrame()

    if not payload:
        return pd.DataFrame()

    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    renamed = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    renamed["Adj Close"] = renamed["Close"]
    renamed["Amount"] = renamed["Close"] * renamed["Volume"]
    return renamed.where(pd.notnull(renamed), None)


def fetch_ohlcv(
    ticker: str,
    start_date: str,
    fmp_api_key: str | None = None,
    allow_yahoo_fallback: bool = True,
) -> pd.DataFrame:
    """종목 시세를 읽는다."""
    if fmp_api_key:
        frame = _fetch_fmp_ohlcv(ticker, start_date, fmp_api_key)
        if not frame.empty:
            return frame

    if not allow_yahoo_fallback:
        return pd.DataFrame()

    frame = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if frame.empty:
        return pd.DataFrame()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    if "Close" in frame.columns and "Volume" in frame.columns:
        frame["Amount"] = frame["Close"] * frame["Volume"]
    else:
        frame["Amount"] = 0

    frame.replace([np.inf, -np.inf], None, inplace=True)
    return frame.where(pd.notnull(frame), None)


def attach_trailing_valuation(price_frame: pd.DataFrame, fundamentals_frame: pd.DataFrame) -> pd.DataFrame:
    """과거 시점 기준 PER, PBR을 price_frame에 붙인다."""
    if price_frame.empty:
        return price_frame

    result = price_frame.copy()
    if fundamentals_frame.empty:
        result["per"] = None
        result["pbr"] = None
        return result

    fund_frame = fundamentals_frame.copy()
    fund_frame["date"] = pd.to_datetime(fund_frame["date"])
    for column in ("equity", "shares_issued", "eps"):
        fund_frame[column] = pd.to_numeric(fund_frame[column], errors="coerce")
    fund_frame = fund_frame.sort_values("date")

    temp = result.copy()
    temp["price_date"] = pd.to_datetime(temp.index)
    merged = pd.merge_asof(
        temp.sort_values("price_date"),
        fund_frame,
        left_on="price_date",
        right_on="date",
        direction="backward",
    )
    merged.index = temp.index

    merged["per"] = merged["Close"] / merged["eps"]
    merged.loc[merged["eps"] <= 0, "per"] = None

    bps = merged["equity"] / merged["shares_issued"]
    merged["pbr"] = merged["Close"] / bps
    merged.loc[(merged["shares_issued"] <= 0) | (bps <= 0), "pbr"] = None

    result["per"] = merged["per"]
    result["pbr"] = merged["pbr"]
    return result.where(pd.notnull(result), None)
