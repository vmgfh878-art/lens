from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd
import yfinance as yf

from backend.collector.errors import SourceLimitReachedError
from backend.collector.sources.eodhd_prices import _fetch_eodhd_ohlcv
from backend.collector.sources.price_contract import prepare_provider_price_frame
from backend.collector.sources.yf_common import prepare_yfinance


prepare_yfinance()


class MarketDataProvider(Protocol):
    name: str

    def fetch_daily(
        self,
        ticker: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class MarketDataFetchResult:
    ticker: str
    requested_provider: str
    provider: str | None
    frame: pd.DataFrame
    fallback_provider: str | None = None
    fallback_used: bool = False
    errors: list[str] = field(default_factory=list)


def normalize_provider_name(provider_name: str | None) -> str:
    if not provider_name:
        return "eodhd"
    normalized = provider_name.strip().lower().replace("-", "_")
    aliases = {
        "yf": "yfinance",
        "yahoo": "yfinance",
        "yahoo_finance": "yfinance",
        "eod": "eodhd",
    }
    return aliases.get(normalized, normalized)


def provider_adjustment_policy(provider_name: str | None) -> str:
    normalized = normalize_provider_name(provider_name)
    policies = {
        "yfinance": "yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc",
        "eodhd": "eodhd_raw_ohlc_adjusted_close_factor_v3_adjusted_ohlc",
    }
    return policies.get(normalized, f"{normalized}_adjusted_close_factor_v3_adjusted_ohlc")


def _filter_end_date(frame: pd.DataFrame, end_date: str | None) -> pd.DataFrame:
    if frame.empty or not end_date:
        return frame
    cutoff = pd.to_datetime(end_date)
    return frame[pd.to_datetime(frame.index) <= cutoff].copy()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume", "Amount"])


class EodhdPriceProvider:
    name = "eodhd"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def fetch_daily(
        self,
        ticker: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        if not self.api_key or not start_date:
            return _empty_frame()
        frame = _fetch_eodhd_ohlcv(ticker, start_date, self.api_key)
        frame = _filter_end_date(frame, end_date)
        if frame.empty:
            return frame
        frame.attrs["market_data_provider"] = self.name
        return prepare_provider_price_frame(frame)


class YFinancePriceProvider:
    name = "yfinance"

    def fetch_daily(
        self,
        ticker: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, object] = {
            "progress": False,
            "auto_adjust": False,
            "actions": False,
            "threads": False,
        }
        if period:
            kwargs["period"] = period
        else:
            if start_date:
                kwargs["start"] = start_date
            if end_date:
                kwargs["end"] = (pd.to_datetime(end_date).date() + timedelta(days=1)).isoformat()

        frame = yf.download(ticker, **kwargs)
        if frame.empty:
            return _empty_frame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        required_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if any(column not in frame.columns for column in required_columns):
            return _empty_frame()

        normalized = frame[required_columns].copy()
        normalized["Amount"] = pd.to_numeric(normalized["Close"], errors="coerce") * pd.to_numeric(
            normalized["Volume"],
            errors="coerce",
        )
        normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
        normalized = prepare_provider_price_frame(normalized)
        normalized.attrs["market_data_provider"] = self.name
        return normalized


def get_market_data_provider(provider_name: str, *, eodhd_api_key: str | None = None) -> MarketDataProvider:
    normalized = normalize_provider_name(provider_name)
    if normalized == "eodhd":
        return EodhdPriceProvider(eodhd_api_key)
    if normalized == "yfinance":
        return YFinancePriceProvider()
    raise ValueError(f"지원하지 않는 market data provider입니다: {provider_name}")


def fetch_market_data(
    ticker: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
    provider_name: str = "eodhd",
    fallback_provider_name: str | None = None,
    eodhd_api_key: str | None = None,
) -> MarketDataFetchResult:
    requested_provider = normalize_provider_name(provider_name)
    fallback_provider = normalize_provider_name(fallback_provider_name) if fallback_provider_name else None
    errors: list[str] = []

    for index, candidate in enumerate([requested_provider, fallback_provider]):
        if candidate is None:
            continue
        try:
            provider = get_market_data_provider(candidate, eodhd_api_key=eodhd_api_key)
            frame = provider.fetch_daily(ticker, start_date=start_date, end_date=end_date, period=period)
        except SourceLimitReachedError:
            raise
        except Exception as exc:
            errors.append(f"{candidate}:{type(exc).__name__}:{exc}")
            continue

        if not frame.empty:
            frame.attrs["market_data_provider"] = provider.name
            return MarketDataFetchResult(
                ticker=ticker,
                requested_provider=requested_provider,
                provider=provider.name,
                frame=frame,
                fallback_provider=fallback_provider,
                fallback_used=index > 0,
                errors=errors,
            )
        errors.append(f"{candidate}:empty")

    return MarketDataFetchResult(
        ticker=ticker,
        requested_provider=requested_provider,
        provider=None,
        frame=_empty_frame(),
        fallback_provider=fallback_provider,
        fallback_used=False,
        errors=errors,
    )


def fetch_many_market_data(
    tickers: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
    provider_name: str = "eodhd",
    fallback_provider_name: str | None = None,
    eodhd_api_key: str | None = None,
) -> dict[str, MarketDataFetchResult]:
    return {
        ticker: fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            period=period,
            provider_name=provider_name,
            fallback_provider_name=fallback_provider_name,
            eodhd_api_key=eodhd_api_key,
        )
        for ticker in tickers
    }
