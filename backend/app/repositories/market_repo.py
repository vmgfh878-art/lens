from __future__ import annotations

import pandas as pd
import structlog

from app.config import get_market_config
from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase
from app.repositories.rest_paging import paged_select

try:
    from collector.repositories.local_snapshots import local_snapshots_required, read_snapshot_frame
except ModuleNotFoundError:
    from backend.collector.repositories.local_snapshots import (
        local_snapshots_required,
        read_snapshot_frame,
    )


logger = structlog.get_logger(__name__)

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
        logger.warning(
            "market data source/provider가 명시되지 않아 MARKET_DATA_PROVIDER 기본값을 사용합니다."
        )
    return _normalize_provider_name(get_market_config().market_data_provider)


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
        normalized["date"] = normalized["date"].map(
            lambda value: str(value)[:10] if value is not None else None
        )
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
        raise UpstreamUnavailableError(
            "로컬 snapshot 모드에서는 가격 API가 Supabase price_data를 조회하지 않습니다."
        )
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
        logger.warning("fetch_price_rows '%s' 실패", ticker, exc_info=exc)
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
        raise UpstreamUnavailableError(
            "로컬 snapshot 모드에서는 보조지표 API가 Supabase indicators를 조회하지 않습니다."
        )

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
                if (
                    missing_column
                    and missing_column in selected_columns
                    and missing_column != "date"
                ):
                    selected_columns.remove(missing_column)
                    missing_columns.add(missing_column)
                    continue
                raise

        for column in missing_columns:
            for row in rows:
                row[column] = None
        # CP254 — indicators.volume 컬럼이 채워져 있으면(서빙 thin 적재) 2차
        # price_data 쿼리를 생략 (egress 절반). 전부 None(legacy 스키마)일 때만 merge.
        if rows and all(row.get("volume") is None for row in rows):
            _merge_indicator_volume(
                ticker=ticker, timeframe=timeframe, rows=rows, provider=provider
            )
        for row in rows:
            for column in INDICATOR_COLUMN_NAMES:
                row.setdefault(column, None)
        return list(reversed(rows))
    except ConfigError:
        raise
    except Exception as exc:
        logger.warning(
            "fetch_indicator_rows '%s' timeframe '%s' 실패", ticker, timeframe, exc_info=exc
        )
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


def _merge_indicator_volume(
    *, ticker: str, timeframe: str, rows: list[dict], provider: str
) -> None:
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
    except Exception as exc:  # noqa: BLE001 — volume merge 는 best-effort. 실패해도 indicator 응답은 유효.
        logger.debug("_merge_indicator_volume best-effort 실패", exc_info=exc)
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


def _query_stock_table(client, table: str, *, search: str | None, limit: int) -> list[dict]:
    # CP254 — prefix 검색을 서버사이드 ilike 로 (limit*50 클라이언트 스캔 제거).
    # 로컬 모드(fetch_stocks_local)의 startswith 와 동일 의미.
    query = client.table(table).select(STOCK_COLUMNS).order("ticker")
    if search:
        query = query.ilike("ticker", f"{search.strip().upper()}%")
    return query.limit(limit).execute().data or []


def _fetch_stock_info_rows(client, *, search: str | None, limit: int) -> list[dict]:
    # CP254 — 1순위 serving_stocks (큐레이션 100, /stocks 목록 정본). 테이블이
    # 없는 legacy DB 에서만 stock_info 로 폴백 (stock_info 는 FK placeholder 포함
    # 유니버스라 sector NULL 행이 섞일 수 있음).
    try:
        rows = _query_stock_table(client, "serving_stocks", search=search, limit=limit)
        return _filter_stock_rows(rows, search=search, limit=limit)
    except Exception as exc:  # noqa: BLE001 — 테이블 부재(legacy)면 stock_info 로
        logger.debug("serving_stocks 조회 실패, stock_info 폴백", exc_info=exc)
    rows = _query_stock_table(client, "stock_info", search=search, limit=limit)
    return _filter_stock_rows(rows, search=search, limit=limit)


def _fetch_stock_info_rows_via_local(*, search: str | None, limit: int) -> list[dict] | None:
    try:
        frame = read_snapshot_frame("stock_info", columns=STOCK_COLUMN_NAMES, order_by="ticker")
    except FileNotFoundError:
        frame = None
    if frame is None:
        return None
    return _filter_stock_rows(frame.to_dict(orient="records"), search=search, limit=limit)


def _fetch_price_ticker_fallback_via_local(
    *, search: str | None, limit: int, provider: str
) -> list[dict] | None:
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


def _fetch_price_ticker_fallback(
    client, *, search: str | None, limit: int, provider: str
) -> list[dict]:
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

    # CP254 — limit*50 증식 제거: 고정 1000행 캡 + prefix 서버 필터.
    # (price_data 는 ticker 당 수백 행이라 distinct 가 안 되는 degraded 폴백 —
    # stock_info 적재 후에는 사실상 미사용 경로.)
    scan_limit = 1000
    query = (
        client.table("price_data").select(PRICE_TICKER_COLUMNS).order("ticker").limit(scan_limit)
    )
    if search:
        query = query.ilike("ticker", f"{search.strip().upper()}%")
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
    local_fallback = _fetch_price_ticker_fallback_via_local(
        search=search, limit=limit, provider=provider
    )
    if local_fallback:
        return local_fallback
    if local_snapshots_required():
        raise UpstreamUnavailableError(
            "로컬 snapshot 모드에서는 종목 검색 API가 Supabase를 조회하지 않습니다."
        )
    try:
        client = get_supabase()
        try:
            rows = _normalize_stock_rows(_fetch_stock_info_rows(client, search=search, limit=limit))
        except Exception as exc:  # noqa: BLE001 — stock_info 1차 폴백. 실패 시 price_data 폴백으로 전환.
            logger.debug("stock_info 1차 폴백 실패, price_data 진행", exc_info=exc)
            rows = []
        if rows:
            return rows
        return _fetch_price_ticker_fallback(client, search=search, limit=limit, provider=provider)
    except ConfigError:
        raise
    except Exception as exc:
        logger.warning("fetch_stocks search='%s' 실패", search, exc_info=exc)
        raise UpstreamUnavailableError("종목 목록을 조회할 수 없습니다.") from exc


# ---------------- 전략 scan 용 cross-sectional 입력 (CP254 P3) ----------------
# 전종목 × 필요 컬럼만 SELECT (전 컬럼 dump 금지). PostgREST max_rows=1000 을
# rest_paging 으로 이어 붙인다. 호출 빈도는 strategy_scan 의 lru_cache 가 억제.

SCAN_PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"]
SCAN_INDICATOR_COLUMNS = [
    "ticker",
    "date",
    "rsi",
    "macd_ratio",
    "ma_5_ratio",
    "ma_20_ratio",
    "ma_60_ratio",
    "bb_position",
]


def fetch_price_frame_for_scan(*, source: str = "eodhd") -> pd.DataFrame:
    """전략 scan base 입력 — price_data 전종목 (OHLCV + adjusted_close)."""
    try:
        client = get_supabase()

        def build_query():
            query = (
                client.table("price_data")
                .select(", ".join(SCAN_PRICE_COLUMNS))
                .order("ticker")
                .order("date")
            )
            return _apply_source_filter(query, _normalize_provider_name(source))

        rows = paged_select(build_query)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_price_frame_for_scan 실패", exc_info=exc)
        raise UpstreamUnavailableError("전략 scan 가격 데이터를 조회할 수 없습니다.") from exc
    return pd.DataFrame(rows, columns=SCAN_PRICE_COLUMNS)


def fetch_indicator_frame_for_scan(*, source: str = "eodhd") -> pd.DataFrame:
    """전략 scan base 입력 — indicators 1D 전종목 (필요 6지표 + rsi)."""
    try:
        client = get_supabase()

        def build_query():
            query = (
                client.table("indicators")
                .select(", ".join(SCAN_INDICATOR_COLUMNS))
                .eq("timeframe", "1D")
                .order("ticker")
                .order("date")
            )
            return _apply_source_filter(query, _normalize_provider_name(source))

        rows = paged_select(build_query)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_indicator_frame_for_scan 실패", exc_info=exc)
        raise UpstreamUnavailableError("전략 scan 지표 데이터를 조회할 수 없습니다.") from exc
    return pd.DataFrame(rows, columns=SCAN_INDICATOR_COLUMNS)


def fetch_sector_frame() -> pd.DataFrame:
    """전략 scan sector map 입력 — serving_stocks (ticker, sector) 만.

    stock_info 가 아니라 serving_stocks 인 이유: stock_info 는 FK placeholder
    (sector NULL ~400행) 포함 유니버스라 sector map 이 'Unknown' 으로 오염된다.
    로컬 모드(market_stock_info.parquet 100행)와 동일 결과를 위해 큐레이션 테이블 사용.
    """
    try:
        client = get_supabase()

        def build_query():
            return client.table("serving_stocks").select("ticker, sector").order("ticker")

        rows = paged_select(build_query)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_sector_frame 실패", exc_info=exc)
        raise UpstreamUnavailableError("전략 scan 섹터 데이터를 조회할 수 없습니다.") from exc
    return pd.DataFrame(rows, columns=["ticker", "sector"])
