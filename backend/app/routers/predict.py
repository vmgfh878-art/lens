from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.db import get_supabase
from app.services.model_svc import normalize_model_name, normalize_timeframe, resolve_horizon

router = APIRouter()


@router.get("/{ticker}")
def predict(
    ticker: str,
    model: str = Query(default="patchtst", description="Prediction model name."),
    timeframe: str = Query(default="1D", description="Prediction timeframe: 1D, 1W, or 1M."),
    horizon: int | None = Query(default=None, description="Prediction horizon in timeframe units."),
):
    try:
        model_name = normalize_model_name(model)
        normalized_timeframe = normalize_timeframe(timeframe)
        resolved_horizon = resolve_horizon(normalized_timeframe, horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    client = get_supabase()
    result = (
        client.table("predictions")
        .select(
            "ticker, model_name, timeframe, horizon, asof_date, decision_time, "
            "run_id, model_ver, signal, forecast_dates, upper_band_series, "
            "lower_band_series, conservative_series"
        )
        .eq("ticker", ticker.upper())
        .eq("model_name", model_name)
        .eq("timeframe", normalized_timeframe)
        .eq("horizon", resolved_horizon)
        .order("asof_date", desc=True)
        .order("decision_time", desc=True)
        .limit(1)
        .execute()
    )

    rows = result.data or []
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No prediction batch found for ticker={ticker.upper()}, "
                f"model={model_name}, timeframe={normalized_timeframe}, horizon={resolved_horizon}."
            ),
        )

    prediction = rows[0]
    prediction["ticker"] = ticker.upper()
    prediction["forecast_dates"] = prediction.get("forecast_dates") or []
    prediction["upper_band_series"] = prediction.get("upper_band_series") or []
    prediction["lower_band_series"] = prediction.get("lower_band_series") or []
    prediction["conservative_series"] = prediction.get("conservative_series") or []
    return prediction
