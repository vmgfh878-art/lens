from __future__ import annotations

import hashlib

from fastapi import APIRouter, Query, Request, Response

from app.core.http import success_response
from app.schemas.common import ApiResponse, ErrorResponse
from app.schemas.stocks import IndicatorResponseData, PredictionData, PriceResponseData, StockSummary
from app.services.api_service import get_indicator_response_data, get_latest_prediction_data, get_price_response_data, get_stocks

router = APIRouter(prefix="/stocks", tags=["stocks"])


def _build_price_etag(payload: dict) -> str:
    rows = payload.get("data") or []
    latest_date = rows[-1]["date"] if rows else "empty"
    raw = f"{payload['ticker']}|{payload['timeframe']}|{payload.get('start')}|{payload.get('end')}|{latest_date}|{len(rows)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"W/\"{digest}\""


@router.get(
    "",
    response_model=ApiResponse[list[StockSummary]],
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def list_stocks(
    request: Request,
    response: Response,
    search: str | None = Query(default=None, description="티커 검색어"),
    limit: int = Query(default=50, ge=1, le=500, description="반환할 최대 종목 수"),
):
    stocks = get_stocks(search=search, limit=limit)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return success_response(request, stocks, total=len(stocks))


@router.get(
    "/{ticker}/prices",
    response_model=ApiResponse[PriceResponseData],
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_prices(
    request: Request,
    response: Response,
    ticker: str,
    start: str | None = Query(default=None, description="조회 시작일"),
    end: str | None = Query(default=None, description="조회 종료일"),
    timeframe: str = Query(default="1D", description="조회 타임프레임"),
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
    ticker: str,
    timeframe: str = Query(default="1D", description="보조지표 타임프레임"),
    limit: int = Query(default=300, ge=1, le=1000, description="반환할 최대 포인트 수"),
):
    data = get_indicator_response_data(ticker, timeframe=timeframe, limit=limit)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return success_response(request, data)


@router.get(
    "/{ticker}/predictions/latest",
    response_model=ApiResponse[PredictionData],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_latest_prediction(
    request: Request,
    response: Response,
    ticker: str,
    model: str = Query(default="patchtst", description="모델 이름"),
    timeframe: str = Query(default="1D", description="예측 타임프레임"),
    horizon: int | None = Query(default=None, description="예측 horizon"),
    run_id: str | None = Query(default=None, description="AI run ID"),
):
    data = get_latest_prediction_data(ticker, model=model, timeframe=timeframe, horizon=horizon, run_id=run_id)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return success_response(request, data)
