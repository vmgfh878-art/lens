from __future__ import annotations

from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase


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
STOCK_COLUMNS = "ticker, sector, industry, market_cap"
PRICE_TICKER_COLUMNS = "ticker"


def fetch_price_rows(
    ticker: str,
    *,
    start: str,
    end: str | None = None,
) -> list[dict]:
    try:
        client = get_supabase()
        query = (
            client.table("price_data")
            .select(PRICE_COLUMNS)
            .eq("ticker", ticker.upper())
            .gte("date", start)
            .order("date")
        )
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
) -> list[dict]:
    selected_columns = INDICATOR_COLUMN_NAMES.copy()
    missing_columns: set[str] = set()

    try:
        rows = []
        while selected_columns:
            client = get_supabase()
            try:
                rows = (
                    client.table("indicators")
                    .select(", ".join(selected_columns))
                    .eq("ticker", ticker.upper())
                    .eq("timeframe", timeframe)
                    .order("date", desc=True)
                    .limit(limit)
                    .execute()
                    .data
                    or []
                )
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
        _merge_indicator_volume(ticker=ticker, timeframe=timeframe, rows=rows)
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


def _merge_indicator_volume(*, ticker: str, timeframe: str, rows: list[dict]) -> None:
    if timeframe != "1D" or not rows:
        return

    dates = [row.get("date") for row in rows if row.get("date")]
    if not dates:
        return

    try:
        client = get_supabase()
        volume_rows = (
            client.table("price_data")
            .select("date, volume")
            .eq("ticker", ticker.upper())
            .in_("date", dates)
            .execute()
            .data
            or []
        )
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


def _fetch_price_ticker_fallback(client, *, search: str | None, limit: int) -> list[dict]:
    # stock_info가 비어 있거나 일시적으로 조회되지 않을 때도 데모 검색은 가격 데이터 기준으로 동작해야 한다.
    if search:
        exact_rows = (
            client.table("price_data")
            .select(PRICE_TICKER_COLUMNS)
            .eq("ticker", search.upper())
            .order("ticker")
            .limit(limit)
            .execute()
            .data
            or []
        )
        exact_matches = _normalize_stock_rows(exact_rows)
        if exact_matches:
            return exact_matches[:limit]

    scan_limit = max(limit * 50, 1000)
    rows = client.table("price_data").select(PRICE_TICKER_COLUMNS).order("ticker").limit(scan_limit).execute().data or []
    return _filter_stock_rows(rows, search=search, limit=limit)


def fetch_stocks(*, search: str | None = None, limit: int = 50) -> list[dict]:
    try:
        client = get_supabase()
        try:
            rows = _normalize_stock_rows(_fetch_stock_info_rows(client, search=search, limit=limit))
        except Exception:
            rows = []
        if rows:
            return rows
        return _fetch_price_ticker_fallback(client, search=search, limit=limit)
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("종목 목록을 조회할 수 없습니다.") from exc
