from __future__ import annotations

import hashlib

from fastapi import APIRouter, Query, Request, Response

from app.services.api_service import get_price_response_data, get_stocks

router = APIRouter()


def _build_price_etag(payload: dict) -> str:
    rows = payload.get("data") or []
    latest_date = rows[-1]["date"] if rows else "empty"
    raw = f"{payload['ticker']}|{payload['timeframe']}|{payload.get('start')}|{payload.get('end')}|{latest_date}|{len(rows)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"W/\"{digest}\""


@router.get("/{ticker}", deprecated=True)
def get_prices(
    request: Request,
    response: Response,
    ticker: str,
    start: str | None = Query(default=None, description="조회 시작일"),
    end: str | None = Query(default=None, description="조회 종료일"),
    timeframe: str = Query(default="1D", description="가격 타임프레임: 1D, 1W, 1M"),
    limit: int = Query(default=366, ge=1, le=1500, description="반환할 최대 포인트 수"),
):
    payload = get_price_response_data(ticker, start=start, end=end, timeframe=timeframe, limit=limit)
    etag = _build_price_etag(payload)
    headers = {"Cache-Control": "public, max-age=3600", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return payload


@router.get("/", deprecated=True)
def get_tickers(
    search: str | None = Query(default=None, description="티커 검색어"),
    limit: int = Query(default=50, ge=1, le=500, description="반환할 최대 종목 수"),
):
    return {"tickers": get_stocks(search=search, limit=limit)}
