from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.core.exceptions import InvalidRunStatusError, ResourceNotFoundError
from app.repositories.ai_repo import fetch_model_run
from app.repositories.market_repo import (
    fetch_indicator_rows,
    fetch_price_rows,
    fetch_stocks,
    resolve_market_data_provider,
)
from app.repositories.prediction_repo import fetch_latest_prediction, fetch_prediction_by_run, fetch_prediction_history_by_run
from app.services.model_svc import (
    normalize_display_timeframe,
    normalize_model_name,
    normalize_prediction_timeframe,
    resolve_horizon,
)
from app.services.feature_svc import drop_incomplete_resampled_periods


DEFAULT_PRICE_WINDOW_DAYS = 365


def aggregate_prices(rows: list[dict], timeframe: str) -> list[dict]:
    normalized_timeframe = normalize_display_timeframe(timeframe)
    if normalized_timeframe == "1D" or not rows:
        return rows

    frame = pd.DataFrame(rows)
    if frame.empty:
        return []

    frame["date"] = pd.to_datetime(frame["date"])
    latest_daily_date = frame["date"].max()
    frame = frame.sort_values("date").set_index("date")
    rule = "W-FRI" if normalized_timeframe == "1W" else "ME"
    aggregated = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()
    aggregated = drop_incomplete_resampled_periods(
        aggregated,
        normalized_timeframe,
        latest_daily_date=latest_daily_date,
    )
    aggregated["date"] = aggregated["date"].dt.strftime("%Y-%m-%d")
    return aggregated.to_dict(orient="records")


def resolve_price_window(start: str | None, end: str | None) -> tuple[str, str]:
    resolved_end = pd.to_datetime(end).date() if end else date.today()
    resolved_start = pd.to_datetime(start).date() if start else resolved_end - timedelta(days=DEFAULT_PRICE_WINDOW_DAYS)
    if resolved_start > resolved_end:
        raise ValueError("조회 시작일은 종료일보다 늦을 수 없습니다.")
    return resolved_start.isoformat(), resolved_end.isoformat()


def get_stocks(*, search: str | None = None, limit: int = 50) -> list[dict]:
    provider = resolve_market_data_provider()
    return fetch_stocks(search=search, limit=limit, market_data_provider=provider)


def get_price_response_data(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    timeframe: str = "1D",
    limit: int | None = None,
) -> dict:
    normalized_timeframe = normalize_display_timeframe(timeframe)
    resolved_start, resolved_end = resolve_price_window(start, end)
    provider = resolve_market_data_provider()
    rows = fetch_price_rows(ticker, start=resolved_start, end=resolved_end, market_data_provider=provider)
    if not rows:
        raise ResourceNotFoundError(f"종목 '{ticker.upper()}'의 가격 데이터를 찾을 수 없습니다.")

    aggregated = aggregate_prices(rows, normalized_timeframe)
    if limit is not None and limit > 0:
        aggregated = aggregated[-limit:]

    return {
        "ticker": ticker.upper(),
        "timeframe": normalized_timeframe,
        "start": resolved_start,
        "end": resolved_end,
        "data": aggregated,
    }


def get_indicator_response_data(
    ticker: str,
    *,
    timeframe: str = "1D",
    limit: int = 300,
) -> dict:
    normalized_timeframe = normalize_display_timeframe(timeframe)
    provider = resolve_market_data_provider()
    rows = fetch_indicator_rows(
        ticker,
        timeframe=normalized_timeframe,
        limit=limit,
        market_data_provider=provider,
    )
    return {
        "ticker": ticker.upper(),
        "timeframe": normalized_timeframe,
        "data": rows,
    }


def _build_prediction_meta(prediction: dict, model_run: dict | None) -> dict:
    meta = prediction.get("meta") if isinstance(prediction.get("meta"), dict) else {}
    merged = dict(meta)
    if not model_run:
        return merged

    config = model_run.get("config") if isinstance(model_run.get("config"), dict) else {}
    feature_version = model_run.get("feature_version") or config.get("feature_version")
    if feature_version:
        merged.setdefault("feature_contract", feature_version)
        merged.setdefault("feature_contract_version", feature_version)
    for key in (
        "line_model_run_id",
        "band_model_run_id",
        "composition_policy",
        "band_calibration_method",
        "band_calibration_params",
        "prediction_composition_version",
        "role",
        "deprecated_for_phase1_product_contract",
        "indicator_layer_replacement",
    ):
        if config.get(key) is not None:
            merged.setdefault(key, config[key])
    return merged


def _normalize_prediction_payload(prediction: dict, *, ticker: str, model_run: dict | None) -> dict:
    prediction["ticker"] = ticker.upper()
    prediction["forecast_dates"] = prediction.get("forecast_dates") or []
    prediction["upper_band_series"] = prediction.get("upper_band_series") or []
    prediction["lower_band_series"] = prediction.get("lower_band_series") or []
    prediction["line_series"] = prediction.get("line_series") or prediction.get("conservative_series") or []
    prediction["conservative_series"] = prediction.get("conservative_series") or prediction["line_series"]
    prediction["meta"] = _build_prediction_meta(prediction, model_run)
    return prediction


def _fetch_completed_model_run(run_id: str) -> dict:
    model_run = fetch_model_run(run_id)
    if model_run is None:
        raise ResourceNotFoundError(f"run_id={run_id} AI run을 찾을 수 없습니다.")
    run_status = str(model_run.get("status") or "completed")
    if run_status != "completed":
        raise InvalidRunStatusError(
            f"run_id={run_id} status={run_status}: completed 상태의 run 예측만 조회할 수 있습니다.",
            details={"run_id": run_id, "status": run_status},
        )
    return model_run


def get_latest_prediction_data(
    ticker: str,
    *,
    model: str = "patchtst",
    timeframe: str = "1D",
    horizon: int | None = None,
    run_id: str | None = None,
) -> dict:
    model_run = None
    if run_id:
        model_run = _fetch_completed_model_run(run_id)
        prediction = fetch_prediction_by_run(ticker, run_id=run_id)
        model_name = str(model_run.get("model_name") or model).strip().lower()
        normalized_timeframe = normalize_prediction_timeframe(str(model_run.get("timeframe") or timeframe))
        resolved_horizon = int(model_run.get("horizon") or resolve_horizon(normalized_timeframe, horizon))
    else:
        model_name = normalize_model_name(model)
        normalized_timeframe = normalize_prediction_timeframe(timeframe)
        resolved_horizon = resolve_horizon(normalized_timeframe, horizon)

        prediction = fetch_latest_prediction(
            ticker,
            model_name=model_name,
            timeframe=normalized_timeframe,
            horizon=resolved_horizon,
        )
    if prediction is None:
        raise ResourceNotFoundError(
            (
                f"ticker={ticker.upper()}, model={model_name}, "
                f"timeframe={normalized_timeframe}, horizon={resolved_horizon} 조건의 예측 결과를 찾을 수 없습니다."
            )
        )

    return _normalize_prediction_payload(prediction, ticker=ticker, model_run=model_run)


def get_prediction_history_data(
    ticker: str,
    *,
    run_id: str,
    limit: int = 90,
) -> list[dict]:
    model_run = _fetch_completed_model_run(run_id)
    normalized_timeframe = normalize_prediction_timeframe(str(model_run.get("timeframe") or "1D"))
    rows = fetch_prediction_history_by_run(ticker, run_id=run_id, limit=limit)
    return [
        _normalize_prediction_payload(
            {**row, "timeframe": row.get("timeframe") or normalized_timeframe},
            ticker=ticker,
            model_run=model_run,
        )
        for row in rows
    ]
