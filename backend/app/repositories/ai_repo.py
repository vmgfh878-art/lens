from __future__ import annotations

from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase


RUN_COLUMNS = (
    "run_id, wandb_run_id, model_name, timeframe, horizon, val_metrics, "
    "test_metrics, config, checkpoint_path, status, created_at"
)
EVALUATION_COLUMNS = (
    "run_id, ticker, timeframe, asof_date, coverage, avg_band_width, "
    "direction_accuracy, mae, smape"
)
BACKTEST_COLUMNS = (
    "run_id, strategy_name, timeframe, return_pct, mdd, sharpe, win_rate, "
    "profit_factor, num_trades, meta, created_at"
)


def fetch_model_runs(
    *,
    model_name: str | None = None,
    timeframe: str | None = None,
    status: str | None = "completed",
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    try:
        client = get_supabase()
        query = client.table("model_runs").select(RUN_COLUMNS).order("created_at", desc=True)
        if model_name:
            query = query.eq("model_name", model_name)
        if timeframe:
            query = query.eq("timeframe", timeframe)
        if status is not None:
            query = query.eq("status", status)
        end = offset + limit - 1
        return query.range(offset, end).execute().data or []
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 목록을 조회할 수 없습니다.") from exc


def fetch_model_run(run_id: str) -> dict | None:
    try:
        client = get_supabase()
        rows = (
            client.table("model_runs")
            .select(RUN_COLUMNS)
            .eq("run_id", run_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 상세를 조회할 수 없습니다.") from exc


def fetch_run_evaluations(
    run_id: str,
    *,
    ticker: str | None = None,
    timeframe: str | None = None,
    limit: int = 100,
) -> list[dict]:
    try:
        client = get_supabase()
        query = (
            client.table("prediction_evaluations")
            .select(EVALUATION_COLUMNS)
            .eq("run_id", run_id)
            .order("asof_date", desc=True)
        )
        if ticker:
            query = query.eq("ticker", ticker.upper())
        if timeframe:
            query = query.eq("timeframe", timeframe)
        return query.limit(limit).execute().data or []
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 평가 결과를 조회할 수 없습니다.") from exc


def fetch_run_backtests(
    run_id: str,
    *,
    strategy_name: str | None = None,
    timeframe: str | None = None,
    limit: int = 50,
) -> list[dict]:
    try:
        client = get_supabase()
        query = (
            client.table("backtest_results")
            .select(BACKTEST_COLUMNS)
            .eq("run_id", run_id)
            .order("created_at", desc=True)
        )
        if strategy_name:
            query = query.eq("strategy_name", strategy_name)
        if timeframe:
            query = query.eq("timeframe", timeframe)
        return query.limit(limit).execute().data or []
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 백테스트 결과를 조회할 수 없습니다.") from exc
