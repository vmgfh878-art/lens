from __future__ import annotations

import requests
import yfinance as yf

from backend.collector.errors import SourceLimitReachedError
from backend.collector.sources.yf_common import prepare_yfinance

prepare_yfinance()


def _fetch_fmp_stock_info(ticker: str, api_key: str) -> dict | None:
    url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key.strip()}"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 429:
            raise SourceLimitReachedError("FMP", response.text.strip())
        response.raise_for_status()
        payload = response.json()
    except SourceLimitReachedError:
        raise
    except Exception:
        return None

    if isinstance(payload, dict) and "Error Message" in payload:
        message = str(payload.get("Error Message", ""))
        if "Limit Reach" in message:
            raise SourceLimitReachedError("FMP", message)
        return None

    if not payload:
        return None

    row = payload[0]
    sector = row.get("sector")
    industry = row.get("industry")
    market_cap = row.get("marketCap")
    if not sector and not industry:
        return None

    return {
        "ticker": ticker.upper(),
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
    }


def fetch_stock_info(
    ticker: str,
    fmp_api_key: str | None = None,
    allow_yahoo_fallback: bool = True,
) -> dict | None:
    """醫낅ぉ 硫뷀??곗씠?곕? ?쎈뒗??"""
    if fmp_api_key:
        profile = _fetch_fmp_stock_info(ticker, fmp_api_key)
        if profile is not None:
            return profile

    if not allow_yahoo_fallback:
        return None

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None

    sector = info.get("sector")
    industry = info.get("industry")
    market_cap = info.get("marketCap")

    if not sector and not industry:
        return None

    return {
        "ticker": ticker.upper(),
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
    }

