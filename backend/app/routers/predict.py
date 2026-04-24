from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.exceptions import TimeframeDisabledError
from app.services.api_service import get_latest_prediction_data
from app.services.model_svc import SUPPORTED_AI_TIMEFRAMES

router = APIRouter()


@router.get("/{ticker}", deprecated=True)
def predict(
    ticker: str,
    model: str = Query(default="patchtst", description="Prediction model name."),
    timeframe: str = Query(default="1D", description="Prediction timeframe: 1D, 1W, or 1M."),
    horizon: int | None = Query(default=None, description="Prediction horizon in timeframe units."),
):
    try:
        return get_latest_prediction_data(ticker, model=model, timeframe=timeframe, horizon=horizon)
    except TimeframeDisabledError:
        return {
            "status": "timeframe_disabled",
            "supported": list(SUPPORTED_AI_TIMEFRAMES),
        }
