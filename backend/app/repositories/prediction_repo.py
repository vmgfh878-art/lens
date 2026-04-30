from __future__ import annotations

from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase


PREDICTION_COLUMNS = (
    "ticker, model_name, timeframe, horizon, asof_date, decision_time, "
    "run_id, model_ver, signal, forecast_dates, upper_band_series, "
    "lower_band_series, conservative_series, line_series, band_quantile_low, band_quantile_high, meta"
)


def fetch_latest_prediction(
    ticker: str,
    *,
    model_name: str,
    timeframe: str,
    horizon: int,
) -> dict | None:
    try:
        client = get_supabase()
        rows = (
            client.table("predictions")
            .select(PREDICTION_COLUMNS)
            .eq("ticker", ticker.upper())
            .eq("model_name", model_name)
            .eq("timeframe", timeframe)
            .eq("horizon", horizon)
            .order("asof_date", desc=True)
            .order("decision_time", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("예측 결과를 조회할 수 없습니다.") from exc


def fetch_prediction_by_run(
    ticker: str,
    *,
    run_id: str,
) -> dict | None:
    try:
        client = get_supabase()
        rows = (
            client.table("predictions")
            .select(PREDICTION_COLUMNS)
            .eq("ticker", ticker.upper())
            .eq("run_id", run_id)
            .order("asof_date", desc=True)
            .order("decision_time", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("run 기준 예측 결과를 조회할 수 없습니다.") from exc
