from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from backend.collector.errors import SourceLimitReachedError


def _to_eodhd_symbol(ticker: str) -> str:
    """誘멸뎅 二쇱떇 ?곗빱瑜?EODHD ?щ낵 ?뺤떇?쇰줈 蹂?섑븳??"""
    normalized = ticker.strip().upper()
    if "." in normalized and not normalized.endswith((".US", ".NYSE", ".NASDAQ", ".BATS", ".AMEX")):
        normalized = normalized.replace(".", "-")
    if "." in normalized:
        return normalized
    return f"{normalized}.US"


def _fetch_eodhd_ohlcv(ticker: str, start_date: str, api_key: str) -> pd.DataFrame:
    symbol = _to_eodhd_symbol(ticker)
    url = f"https://eodhd.com/api/eod/{symbol}"
    try:
        response = requests.get(
            url,
            params={
                "api_token": api_key.strip(),
                "fmt": "json",
                "from": start_date,
            },
            timeout=30,
        )
        if response.status_code == 429:
            raise SourceLimitReachedError("EODHD", response.text.strip())
        response.raise_for_status()
        payload = response.json()
    except SourceLimitReachedError:
        raise
    except Exception:
        return pd.DataFrame()

    if isinstance(payload, dict):
        message = str(payload.get("message") or payload.get("error") or "").strip()
        if "limit" in message.lower():
            raise SourceLimitReachedError("EODHD", message)
        return pd.DataFrame()

    if not payload:
        return pd.DataFrame()

    frame = pd.DataFrame(payload)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    renamed = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "Adj Close",
            "volume": "Volume",
        }
    )
    if "Adj Close" not in renamed.columns:
        renamed["Adj Close"] = renamed["Close"]
    renamed["Amount"] = renamed["Close"] * renamed["Volume"]
    renamed.replace([np.inf, -np.inf], None, inplace=True)
    return renamed.where(pd.notnull(renamed), None)


def fetch_ohlcv(
    ticker: str,
    start_date: str,
    eodhd_api_key: str | None = None,
    allow_yahoo_fallback: bool = False,
) -> pd.DataFrame:
    """醫낅ぉ蹂?OHLCV瑜??쎈뒗?? 湲곕낯 ?뚯뒪??EODHD??"""
    from backend.collector.sources.market_data_providers import fetch_market_data

    result = fetch_market_data(
        ticker,
        start_date=start_date,
        provider_name="eodhd",
        fallback_provider_name="yfinance" if allow_yahoo_fallback else None,
        eodhd_api_key=eodhd_api_key,
    )
    return result.frame


def attach_trailing_valuation(price_frame: pd.DataFrame, fundamentals_frame: pd.DataFrame) -> pd.DataFrame:
    """怨쇨굅 ?쒖젏 湲곗? PER, PBR??price_frame??遺숈씤??"""
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

