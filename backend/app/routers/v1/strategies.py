from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from app.core.http import success_response
from app.services.strategy_backtest_svc import (
    STRATEGIES,
    get_strategy_backtest,
    get_strategy_scan,
)

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("")
def list_strategies(request: Request, response: Response):
    """프론트 백테스트 화면에서 사용할 단일 티커 long/cash 전략 목록."""
    response.headers["Cache-Control"] = "public, max-age=3600"
    return success_response(
        request,
        [
            {
                "id": rule.id,
                "label": rule.label,
                "uses_line": rule.uses_line,
                "uses_band": rule.uses_band,
                "uses_ai": rule.uses_ai,
                "contract": "single_ticker_long_cash",
            }
            for rule in STRATEGIES.values()
        ],
    )


@router.get("/{strategy_id}/scan")
def scan_strategy(
    request: Request,
    response: Response,
    strategy_id: str,
    limit: int = Query(default=500, ge=1, le=500),
):
    """500티커 로컬 캐시 기준 최신 단일 티커 전략 신호 카드."""
    data = get_strategy_scan(strategy_id, limit=limit)
    response.headers["Cache-Control"] = "public, max-age=300"
    return success_response(request, data, total=len(data["cards"]))


@router.get("/{strategy_id}/backtest/{ticker}")
def backtest_strategy_ticker(
    request: Request,
    response: Response,
    strategy_id: str,
    ticker: str,
):
    """선택 티커 하나에 대한 long/cash 상세 백테스트."""
    data = get_strategy_backtest(strategy_id, ticker)
    response.headers["Cache-Control"] = "public, max-age=300"
    return success_response(request, data)
