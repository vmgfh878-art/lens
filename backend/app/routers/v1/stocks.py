from __future__ import annotations

import hashlib
from typing import Literal

from app.core.http import success_response
from app.core.validators import SearchStr, TickerStr
from app.schemas.common import ApiResponse, ErrorResponse
from app.schemas.stocks import (
    IndicatorResponseData,
    PriceResponseData,
    ProductPredictionHistoryResponseData,
    StockSummary,
)
from app.services.api_service import (
    get_indicator_response_data,
    get_price_response_data,
    get_stocks,
)
from app.services.product_prediction_history_svc import get_product_prediction_history_data
from fastapi import APIRouter, Query, Request, Response

router = APIRouter(prefix="/stocks", tags=["stocks"])


def _build_price_etag(payload: dict) -> str:
    rows = payload.get("data") or []
    latest_date = rows[-1]["date"] if rows else "empty"
    raw = (
        f"{payload['ticker']}|{payload['timeframe']}"
        f"|{payload.get('start')}|{payload.get('end')}"
        f"|{latest_date}|{len(rows)}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f'W/"{digest}"'


def _build_indicator_etag(payload: dict, *, limit: int) -> str:
    # CP254 P4 — indicators 도 ETag 304 (prices 와 동일 패턴). 최신 date + 행수 기준.
    rows = payload.get("data") or []
    latest_date = rows[-1].get("date") if rows else "empty"
    raw = f"{payload['ticker']}|{payload['timeframe']}|{limit}|{latest_date}|{len(rows)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f'W/"{digest}"'


def _build_product_history_etag(payload: dict, *, params: dict) -> str:
    # CP254 P4 — product-history ETag. 최신 asof + role별 행수 + 쿼리 파라미터 기준.
    param_text = "|".join(f"{key}={params[key]}" for key in sorted(params))
    raw = (
        f"{payload['ticker']}|{payload['timeframe']}|{payload.get('latest_asof_date')}"
        f"|{len(payload.get('line_history') or [])}|{len(payload.get('band_history') or [])}"
        f"|{param_text}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f'W/"{digest}"'


@router.get(
    "",
    response_model=ApiResponse[list[StockSummary]],
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def list_stocks(
    request: Request,
    response: Response,
    search: SearchStr | None = Query(default=None, description="티커 검색어"),
    limit: int = Query(default=50, ge=1, le=500, description="반환할 최대 종목 수"),
):
    stocks = get_stocks(search=search, limit=limit)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return success_response(request, stocks, total=len(stocks))


@router.get(
    "/{ticker}/prices",
    response_model=ApiResponse[PriceResponseData],
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def get_prices(
    request: Request,
    response: Response,
    ticker: TickerStr,
    start: str | None = Query(default=None, description="조회 시작일"),
    end: str | None = Query(default=None, description="조회 종료일"),
    timeframe: Literal["1D", "1W"] = Query(default="1D", description="조회 타임프레임"),
    limit: int = Query(default=366, ge=1, le=1500, description="반환할 최대 포인트 수"),
):
    data = get_price_response_data(ticker, start=start, end=end, timeframe=timeframe, limit=limit)
    etag = _build_price_etag(data)
    headers = {"Cache-Control": "public, max-age=3600", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return success_response(request, data)


@router.get(
    "/{ticker}/indicators",
    response_model=ApiResponse[IndicatorResponseData],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_indicators(
    request: Request,
    response: Response,
    ticker: TickerStr,
    timeframe: Literal["1D", "1W"] = Query(default="1D", description="보조지표 타임프레임"),
    limit: int = Query(default=300, ge=1, le=1000, description="반환할 최대 포인트 수"),
):
    data = get_indicator_response_data(ticker, timeframe=timeframe, limit=limit)
    etag = _build_indicator_etag(data, limit=limit)
    headers = {"Cache-Control": "public, max-age=3600", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return success_response(request, data)


@router.get(
    "/{ticker}/predictions/product-history",
    response_model=ApiResponse[ProductPredictionHistoryResponseData],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_product_prediction_history(
    request: Request,
    response: Response,
    ticker: TickerStr,
    timeframe: Literal["1D", "1W"] = Query(
        default="1D", description="제품 rolling history timeframe"
    ),
    roles: str = Query(default="all", description="all, line, band 또는 line,band"),
    run_id: str | None = Query(default=None, description="선택 run_id 필터"),
    limit: int | None = Query(
        default=None, ge=1, le=1500, description="role별 최근 history row 수"
    ),
    # CP254 P4 — 기본 lookback 370일 강제 (프론트 상수 PRODUCT_HISTORY_LOOKBACK_DAYS
    # 와 정합). 미지정 호출이 종목 전체 rolling history(최대 수천 행)를 dump 하던
    # egress 1순위 구멍을 막는다. 전체가 필요하면 lookback_days=1500 명시.
    lookback_days: int | None = Query(
        default=370, ge=1, le=1500, description="최근 N일 history (기본 370)"
    ),
):
    data = get_product_prediction_history_data(
        ticker,
        timeframe=timeframe,
        roles=roles,
        run_id=run_id,
        limit=limit,
        lookback_days=lookback_days,
    )
    etag = _build_product_history_etag(
        data,
        params={
            "roles": roles,
            "run_id": run_id,
            "limit": limit,
            "lookback_days": lookback_days,
        },
    )
    headers = {"Cache-Control": "public, max-age=3600", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    total = len(data["line_history"]) + len(data["band_history"])
    return success_response(request, data, total=total)
