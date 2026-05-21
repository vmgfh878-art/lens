from __future__ import annotations

import logging
import os

from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase
try:
    from collector.repositories.local_snapshots import local_snapshots_required, read_snapshot_frame
except ModuleNotFoundError:
    from backend.collector.repositories.local_snapshots import local_snapshots_required, read_snapshot_frame


logger = logging.getLogger(__name__)

PRICE_COLUMN_NAMES = ["date", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = "date, open, high, low, close, volume"
INDICATOR_COLUMN_NAMES = [
    "date",
    "rsi",
    "macd_ratio",
    "bb_position",
    "ma_5_ratio",
    "ma_20_ratio",
    "ma_60_ratio",
    "vol_change",
    "volume",
    "atr_ratio",
    "regime_label",
]
STOCK_COLUMN_NAMES = ["ticker", "sector", "industry", "market_cap"]
STOCK_COLUMNS = "ticker, sector, industry, market_cap"
PRICE_TICKER_COLUMNS = "ticker"
_PROVIDER_ALIASES = {
    "yf": "yfinance",
    "yahoo": "yfinance",
    "yahoo_finance": "yfinance",
    "eod": "eodhd",
}


def _normalize_provider_name(provider_name: str | None) -> str:
    if not provider_name:
        return "yfinance"
    normalized = provider_name.strip().lower().replace("-", "_")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def resolve_market_data_provider(
    market_data_provider: str | None = None,
    source: str | None = None,
    *,
    warn_if_default: bool = False,
) -> str:
    if source:
        return _normalize_provider_name(source)
    if market_data_provider:
        return _normalize_provider_name(market_data_provider)
    if warn_if_default:
        logger.warning("market data source/provider가 명시되지 않아 MARKET_DATA_PROVIDER 기본값을 사용합니다.")
    return _normalize_provider_name(os.environ.get("MARKET_DATA_PROVIDER", "yfinance"))


def _apply_source_filter(query, provider: str):
    if provider == "eodhd":
        if hasattr(query, "or_"):
            return query.or_("source.eq.eodhd,source.is.null")
        return query
    if not hasattr(query, "eq"):
        return query
    return query.eq("source", provider)


def _filter_local_provider(frame, provider: str):
    if frame is None or frame.empty:
        return frame
    if "source" not in frame.columns:
        return frame if provider == "eodhd" else frame.iloc[0:0]
    source = frame["source"].astype("string")
    if provider == "eodhd":
        return frame[source.isna() | (source.str.lower() == "eodhd")]
    return frame[source.str.lower() == provider]


def _local_rows(frame, columns: list[str]) -> list[dict]:
    if frame is None or frame.empty:
        return []
    normalized = frame.copy()
    if "date" in normalized.columns:
        normalized["date"] = normalized["date"].map(lambda value: str(value)[:10] if value is not None else None)
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[columns].where(normalized[columns].notnull(), None).to_dict(orient="records")


def fetch_price_rows(
    ticker: str,
    *,
    start: str,
    end: str | None = None,
    market_data_provider: str | None = None,
    source: str | None = None,
) -> list[dict]:
    provider = resolve_market_data_provider(
        market_data_provider,
        source,
        warn_if_default=market_data_provider is None and source is None,
    )
    filters: list[tuple[str, str, object]] = [
        ("eq", "ticker", ticker.upper()),
        ("gte", "date", start),
    ]
    if end:
        filters.append(("lte", "date", end))
    local_frame = read_snapshot_frame(
        "price_data",
        columns=[*PRICE_COLUMN_NAMES, "ticker", "source", "provider"],
        filters=filters,
        order_by="date",
        provider=provider,
    )
    if local_frame is not None:
        return _local_rows(_filter_local_provider(local_frame, provider), PRICE_COLUMN_NAMES)
    if local_snapshots_required():
        raise UpstreamUnavailableError("로컬 snapshot 모드에서는 가격 API가 Supabase price_data를 조회하지 않습니다.")
    try:
        client = get_supabase()
        query = (
            client.table("price_data")
            .select(PRICE_COLUMNS)
            .eq("ticker", ticker.upper())
            .gte("date", start)
            .order("date")
        )
        query = _apply_source_filter(query, provider)
        if end:
            query = query.lte("date", end)
        return query.execute().data or []
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("가격 데이터를 조회할 수 없습니다.") from exc


def fetch_indicator_rows(
    ticker: str,
    *,
    timeframe: str,
    limit: int = 300,
    market_data_provider: str | None = None,
    source: str | None = None,
) -> list[dict]:
    provider = resolve_market_data_provider(
        market_data_provider,
        source,
        warn_if_default=market_data_provider is None and source is None,
    )
    selected_columns = INDICATOR_COLUMN_NAMES.copy()
    missing_columns: set[str] = set()
    local_frame = read_snapshot_frame(
        "indicators",
        columns=[*INDICATOR_COLUMN_NAMES, "ticker", "timeframe", "source", "provider"],
        filters=[("eq", "ticker", ticker.upper()), ("eq", "timeframe", timeframe)],
        order_by="date",
        provider=provider,
        timeframe=timeframe,
    )
    if local_frame is not None:
        local_frame = _filter_local_provider(local_frame, provider)
        if not local_frame.empty:
            local_frame = local_frame.sort_values("date").tail(limit)
        return _local_rows(local_frame, INDICATOR_COLUMN_NAMES)
    if local_snapshots_required():
        raise UpstreamUnavailableError("로컬 snapshot 모드에서는 보조지표 API가 Supabase indicators를 조회하지 않습니다.")

    try:
        rows = []
        while selected_columns:
            client = get_supabase()
            try:
                query = (
                    client.table("indicators")
                    .select(", ".join(selected_columns))
                    .eq("ticker", ticker.upper())
                    .eq("timeframe", timeframe)
                    .order("date", desc=True)
                    .limit(limit)
                )
                query = _apply_source_filter(query, provider)
                rows = query.execute().data or []
                break
            except Exception as exc:
                missing_column = _extract_missing_indicator_column(exc)
                if missing_column and missing_column in selected_columns and missing_column != "date":
                    selected_columns.remove(missing_column)
                    missing_columns.add(missing_column)
                    continue
                raise

        for column in missing_columns:
            for row in rows:
                row[column] = None
        _merge_indicator_volume(ticker=ticker, timeframe=timeframe, rows=rows, provider=provider)
        for row in rows:
            for column in INDICATOR_COLUMN_NAMES:
                row.setdefault(column, None)
        return list(reversed(rows))
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("보조지표 데이터를 조회할 수 없습니다.") from exc


def _extract_missing_indicator_column(exc: Exception) -> str | None:
    message = str(exc)
    prefix = "column indicators."
    if prefix not in message or "does not exist" not in message:
        return None
    start = message.find(prefix) + len(prefix)
    end = message.find(" does not exist", start)
    if end <= start:
        return None
    return message[start:end]


def _merge_indicator_volume(*, ticker: str, timeframe: str, rows: list[dict], provider: str) -> None:
    if timeframe != "1D" or not rows:
        return

    dates = [row.get("date") for row in rows if row.get("date")]
    if not dates:
        return

    try:
        client = get_supabase()
        query = (
            client.table("price_data")
            .select("date, volume")
            .eq("ticker", ticker.upper())
            .in_("date", dates)
        )
        query = _apply_source_filter(query, provider)
        volume_rows = query.execute().data or []
    except Exception:
        return

    volume_by_date = {str(row.get("date")): row.get("volume") for row in volume_rows}
    for row in rows:
        row["volume"] = volume_by_date.get(str(row.get("date")))


def _normalize_stock_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    seen: set[str] = set()
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(
            {
                "ticker": ticker,
                "sector": row.get("sector"),
                "industry": row.get("industry"),
                "market_cap": row.get("market_cap"),
            }
        )
    return normalized


def _filter_stock_rows(rows: list[dict], *, search: str | None, limit: int) -> list[dict]:
    normalized = _normalize_stock_rows(rows)
    if search:
        keyword = search.strip().upper()
        normalized = [row for row in normalized if keyword in row["ticker"]]
    return normalized[:limit]


def _fetch_stock_info_rows(client, *, search: str | None, limit: int) -> list[dict]:
    scan_limit = max(limit * 50, 1000) if search else limit
    rows = client.table("stock_info").select(STOCK_COLUMNS).order("ticker").limit(scan_limit).execute().data or []
    return _filter_stock_rows(rows, search=search, limit=limit)


def _fetch_stock_info_rows_via_local(*, search: str | None, limit: int) -> list[dict] | None:
    try:
        frame = read_snapshot_frame("stock_info", columns=STOCK_COLUMN_NAMES, order_by="ticker")
    except FileNotFoundError:
        frame = None
    if frame is None:
        return None
    return _filter_stock_rows(frame.to_dict(orient="records"), search=search, limit=limit)


def _fetch_price_ticker_fallback_via_local(*, search: str | None, limit: int, provider: str) -> list[dict] | None:
    try:
        frame = read_snapshot_frame(
            "price_data",
            columns=["ticker", "source", "provider"],
            order_by="ticker",
            provider=provider,
        )
    except FileNotFoundError:
        frame = None
    if frame is None:
        return None
    frame = _filter_local_provider(frame, provider)
    return _filter_stock_rows(frame.to_dict(orient="records"), search=search, limit=limit)


def _fetch_price_ticker_fallback(client, *, search: str | None, limit: int, provider: str) -> list[dict]:
    # stock_info가 비어 있거나 일시적으로 조회되지 않을 때도 데모 검색은 가격 데이터 기준으로 동작해야 한다.
    if search:
        exact_query = (
            client.table("price_data")
            .select(PRICE_TICKER_COLUMNS)
            .eq("ticker", search.upper())
            .order("ticker")
            .limit(limit)
        )
        exact_query = _apply_source_filter(exact_query, provider)
        exact_rows = exact_query.execute().data or []
        exact_matches = _normalize_stock_rows(exact_rows)
        if exact_matches:
            return exact_matches[:limit]

    scan_limit = max(limit * 50, 1000)
    query = client.table("price_data").select(PRICE_TICKER_COLUMNS).order("ticker").limit(scan_limit)
    query = _apply_source_filter(query, provider)
    rows = query.execute().data or []
    return _filter_stock_rows(rows, search=search, limit=limit)


def fetch_stocks(
    *,
    search: str | None = None,
    limit: int = 50,
    market_data_provider: str | None = None,
    source: str | None = None,
) -> list[dict]:
    provider = resolve_market_data_provider(
        market_data_provider,
        source,
        warn_if_default=market_data_provider is None and source is None,
    )
    local_rows = _fetch_stock_info_rows_via_local(search=search, limit=limit)
    if local_rows:
        return local_rows
    local_fallback = _fetch_price_ticker_fallback_via_local(search=search, limit=limit, provider=provider)
    if local_fallback:
        return local_fallback
    if local_snapshots_required():
        raise UpstreamUnavailableError("로컬 snapshot 모드에서는 종목 검색 API가 Supabase를 조회하지 않습니다.")
    try:
        client = get_supabase()
        try:
            rows = _normalize_stock_rows(_fetch_stock_info_rows(client, search=search, limit=limit))
        except Exception:
            rows = []
        if rows:
            return rows
        return _fetch_price_ticker_fallback(client, search=search, limit=limit, provider=provider)
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("종목 목록을 조회할 수 없습니다.") from exc
