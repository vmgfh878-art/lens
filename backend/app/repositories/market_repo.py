from __future__ import annotations

from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase


PRICE_COLUMNS = "date, open, high, low, close, volume"
STOCK_COLUMNS = "ticker, sector, industry, market_cap"


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


def fetch_stocks(*, search: str | None = None, limit: int = 50) -> list[dict]:
    try:
        client = get_supabase()
        query = client.table("stock_info").select(STOCK_COLUMNS).order("ticker").limit(limit)
        if search:
            query = query.ilike("ticker", f"%{search.upper()}%")
        return query.execute().data or []
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("종목 목록을 조회할 수 없습니다.") from exc
