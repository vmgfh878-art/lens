from __future__ import annotations

import os
from dataclasses import dataclass

from collector.universe import get_default_universe_file


DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "AMD")


@dataclass(frozen=True)
class CollectorSettings:
    """수집 파이프라인 전역 설정."""

    fred_api_key: str | None
    fmp_api_key: str | None
    universe_file: str
    fmp_daily_limit: int
    stock_info_batch_limit: int
    price_batch_limit: int
    default_price_start: str
    price_lookback_days: int
    macro_lookback_days: int
    sector_lookback_days: int
    indicator_lookback_days: int
    breadth_min_tickers: int
    allow_yahoo_fallback: bool
    use_yahoo_fundamentals_baseline: bool
    stock_info_sleep_seconds: float
    fundamentals_sleep_seconds: float
    price_sleep_seconds: float


def get_settings() -> CollectorSettings:
    return CollectorSettings(
        fred_api_key=os.environ.get("FRED_API_KEY"),
        fmp_api_key=os.environ.get("FMP_API_KEY"),
        universe_file=os.environ.get("LENS_UNIVERSE_FILE", str(get_default_universe_file())),
        fmp_daily_limit=int(os.environ.get("FMP_DAILY_LIMIT", "80")),
        stock_info_batch_limit=int(os.environ.get("LENS_STOCK_INFO_BATCH_LIMIT", "80")),
        price_batch_limit=int(os.environ.get("LENS_PRICE_BATCH_LIMIT", "80")),
        default_price_start=os.environ.get("LENS_PRICE_START_DATE", "2015-01-01"),
        price_lookback_days=int(os.environ.get("LENS_PRICE_LOOKBACK_DAYS", "45")),
        macro_lookback_days=int(os.environ.get("LENS_MACRO_LOOKBACK_DAYS", str(365 * 2))),
        sector_lookback_days=int(os.environ.get("LENS_SECTOR_LOOKBACK_DAYS", str(365 * 2))),
        indicator_lookback_days=int(os.environ.get("LENS_INDICATOR_LOOKBACK_DAYS", "2400")),
        breadth_min_tickers=int(os.environ.get("LENS_BREADTH_MIN_TICKERS", "50")),
        allow_yahoo_fallback=os.environ.get("LENS_ALLOW_YAHOO_FALLBACK", "0") == "1",
        use_yahoo_fundamentals_baseline=os.environ.get("LENS_USE_YAHOO_FUNDAMENTALS_BASELINE", "0") == "1",
        stock_info_sleep_seconds=float(os.environ.get("LENS_STOCK_INFO_SLEEP_SECONDS", "0.2")),
        fundamentals_sleep_seconds=float(os.environ.get("LENS_FUNDAMENTALS_SLEEP_SECONDS", "0.5")),
        price_sleep_seconds=float(os.environ.get("LENS_PRICE_SLEEP_SECONDS", "0.3")),
    )


def _normalize_tickers(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = value.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return normalized


def resolve_target_tickers(cli_tickers: list[str] | None, known_tickers: list[str] | None = None) -> list[str]:
    """실행 대상 티커를 결정한다."""
    if cli_tickers:
        return _normalize_tickers(cli_tickers)

    env_tickers = os.environ.get("LENS_TICKERS")
    if env_tickers:
        return _normalize_tickers(env_tickers.split(","))

    if known_tickers:
        normalized = _normalize_tickers(known_tickers)
        if normalized:
            return normalized

    return list(DEFAULT_TICKERS)
