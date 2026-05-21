from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import os
import time
from typing import Protocol

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backend.collector.errors import SourceLimitReachedError
from backend.collector.sources.eodhd_prices import _fetch_eodhd_ohlcv
from backend.collector.sources.price_contract import prepare_provider_price_frame
from backend.collector.sources.yf_common import prepare_yfinance
from backend.collector.utils.network import sanitize_proxy_env


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


def _yahoo_chart_headers(browser_like: bool = True) -> dict[str, str]:
    if not browser_like:
        return {}
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    }


def _date_to_unix_seconds(value: str | None) -> int | None:
    if not value:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())


def _build_yahoo_chart_params(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
    }
    if period:
        params["range"] = period
        return params
    if start_date:
        params["period1"] = _date_to_unix_seconds(start_date)
    if end_date:
        exclusive_end = (pd.to_datetime(end_date).date() + timedelta(days=1)).isoformat()
        params["period2"] = _date_to_unix_seconds(exclusive_end)
    if "period1" not in params or "period2" not in params:
        params["range"] = "10d"
    return params


def _parse_yahoo_chart_json(ticker: str, payload: dict) -> pd.DataFrame:
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        raise ValueError("yahoo_chart_missing_chart")
    error = chart.get("error")
    if error:
        raise ValueError(f"yahoo_chart_error:{error}")
    results = chart.get("result") or []
    if not results:
        return _empty_frame()
    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or []
    adjcloses = indicators.get("adjclose") or []
    if not timestamps or not quotes:
        return _empty_frame()
    quote = quotes[0]
    adjclose = adjcloses[0].get("adjclose") if adjcloses else None
    if adjclose is None:
        raise ValueError("yahoo_chart_missing_adjclose")

    dates = pd.to_datetime(timestamps, unit="s", utc=True)
    try:
        dates = dates.tz_convert("America/New_York")
    except TypeError:
        dates = dates.tz_localize("UTC").tz_convert("America/New_York")
    index = dates.tz_localize(None).normalize()
    frame = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Adj Close": adjclose,
            "Volume": quote.get("volume", []),
        },
        index=index,
    )
    frame["Amount"] = pd.to_numeric(frame["Close"], errors="coerce") * pd.to_numeric(frame["Volume"], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"], how="any")
    frame = prepare_provider_price_frame(frame)
    frame.attrs["market_data_provider"] = "yfinance"
    frame.attrs["fetch_method"] = "yahoo_chart"
    frame.attrs["ticker"] = ticker
    return frame


def fetch_yahoo_chart_frame(
    ticker: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
    session: requests.Session | None = None,
    timeout_seconds: float | None = None,
    browser_like_headers: bool = True,
) -> pd.DataFrame:
    timeout = float(os.environ.get("YAHOO_CHART_TIMEOUT_SECONDS", "15")) if timeout_seconds is None else timeout_seconds
    params = _build_yahoo_chart_params(start_date=start_date, end_date=end_date, period=period)
    active_session = session or requests.Session()
    active_session.trust_env = False
    errors: list[str] = []
    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        url = f"https://{host}/v8/finance/chart/{ticker}"
        try:
            response = active_session.get(
                url,
                params=params,
                headers=_yahoo_chart_headers(browser_like_headers),
                timeout=timeout,
            )
        except Exception as exc:
            errors.append(f"{host}:{type(exc).__name__}:{exc}")
            continue
        content_type = response.headers.get("content-type", "")
        if response.status_code == 429:
            raise SourceLimitReachedError("yahoo_chart", f"yahoo_chart_429:{ticker}:{host}", reason="rate")
        if response.status_code >= 400:
            errors.append(f"{host}:http_{response.status_code}:{content_type}")
            continue
        try:
            payload = response.json()
        except ValueError as exc:
            preview = response.text[:120].replace("\n", " ")
            errors.append(f"{host}:json_decode:{content_type}:{preview}:{exc}")
            continue
        frame = _parse_yahoo_chart_json(ticker, payload)
        if not frame.empty:
            frame.attrs["fetch_method"] = "yahoo_chart"
            frame.attrs["yahoo_chart_host"] = host
            frame.attrs["yahoo_chart_params"] = params
            return frame
        errors.append(f"{host}:empty_chart")
    empty = _empty_frame()
    empty.attrs["fetch_method"] = "yahoo_chart"
    empty.attrs["yahoo_chart_errors"] = errors
    return empty


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

    def __init__(
        self,
        max_retries: int | None = None,
        retry_sleep_seconds: float | None = None,
        enable_direct_chart_fallback: bool | None = None,
    ) -> None:
        self.max_retries = int(os.environ.get("YFINANCE_MAX_RETRIES", "0")) if max_retries is None else max_retries
        self.retry_sleep_seconds = (
            float(os.environ.get("YFINANCE_RETRY_SLEEP_SECONDS", "1.0"))
            if retry_sleep_seconds is None
            else retry_sleep_seconds
        )
        self.enable_direct_chart_fallback = (
            os.environ.get("YFINANCE_DIRECT_CHART_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}
            if enable_direct_chart_fallback is None
            else enable_direct_chart_fallback
        )
        self.prefer_direct_chart = os.environ.get("YFINANCE_FETCH_METHOD", "direct_chart").strip().lower() in {
            "direct_chart",
            "yahoo_chart",
        }
        sanitize_proxy_env()

    def _fetch_direct_chart(
        self,
        ticker: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        direct_frame = fetch_yahoo_chart_frame(
            ticker,
            start_date=start_date,
            end_date=end_date,
            period=period,
        )
        if not direct_frame.empty:
            direct_frame.attrs["market_data_provider"] = self.name
            direct_frame.attrs["fetch_method"] = "yahoo_chart"
        return direct_frame

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

        if self.enable_direct_chart_fallback and self.prefer_direct_chart:
            direct_first = self._fetch_direct_chart(ticker, start_date=start_date, end_date=end_date, period=period)
            if not direct_first.empty:
                return direct_first

        frame = _empty_frame()
        last_error: BaseException | None = None
        sanitize_proxy_env()
        for attempt in range(self.max_retries + 1):
            try:
                frame = yf.download(ticker, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds)
                    continue
                break
            if not frame.empty or attempt >= self.max_retries:
                break
            time.sleep(self.retry_sleep_seconds)
        if frame.empty:
            if self.enable_direct_chart_fallback:
                direct_frame = self._fetch_direct_chart(ticker, start_date=start_date, end_date=end_date, period=period)
                if not direct_frame.empty:
                    return direct_frame
            if last_error is not None:
                raise last_error
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
        normalized.attrs["fetch_method"] = "yf_download"
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
