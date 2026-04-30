from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Query, Request

from app.core.exceptions import ResourceNotFoundError
from app.core.http import success_response
from app.repositories.ai_repo import (
    fetch_model_run,
    fetch_model_runs,
    fetch_run_backtests,
    fetch_run_evaluations,
)
from app.schemas.ai import BacktestSummary, EvaluationSummary, RunDetail, RunSummary
from app.schemas.common import ApiResponse, ErrorResponse

router = APIRouter(prefix="/ai", tags=["ai"])

CONFIG_SUMMARY_KEYS = (
    "target",
    "seq_len",
    "patch_len",
    "stride",
    "d_model",
    "n_heads",
    "n_layers",
    "dropout",
    "lr",
    "weight_decay",
    "batch_size",
    "epochs",
    "seed",
    "ci_aggregate",
    "ci_target_fast",
    "band_mode",
    "line_model_run_id",
    "band_model_run_id",
    "composition_policy",
    "band_calibration_method",
    "band_calibration_params",
    "prediction_composition_version",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _first_value(row: dict[str, Any], key: str, *fallbacks: dict[str, Any]) -> Any:
    value = row.get(key)
    if value is not None:
        return _json_safe(value)
    for fallback in fallbacks:
        value = fallback.get(key)
        if value is not None:
            return _json_safe(value)
    return None


def _build_run_summary(row: dict[str, Any]) -> dict[str, Any]:
    config = _as_dict(row.get("config"))
    val_metrics = _as_dict(row.get("val_metrics"))
    test_metrics = _as_dict(row.get("test_metrics"))
    best_val_total = _first_value(row, "best_val_total", val_metrics, test_metrics, config)
    return {
        "run_id": row["run_id"],
        "status": _first_value(row, "status"),
        "model_name": _first_value(row, "model_name"),
        "timeframe": _first_value(row, "timeframe"),
        "horizon": _first_value(row, "horizon"),
        "created_at": _first_value(row, "created_at"),
        "model_ver": _first_value(row, "model_ver", config, val_metrics, test_metrics),
        "feature_version": _first_value(row, "feature_version", config),
        "band_mode": _first_value(row, "band_mode", config),
        "checkpoint_path": _first_value(row, "checkpoint_path", config),
        "best_epoch": _first_value(row, "best_epoch", val_metrics, test_metrics, config),
        "best_val_total": best_val_total if best_val_total is not None else _first_value(row, "best_val_loss", config),
        "line_target_type": _first_value(row, "line_target_type", config),
        "band_target_type": _first_value(row, "band_target_type", config),
    }


def _build_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(config.get(key)) for key in CONFIG_SUMMARY_KEYS}


def _build_run_detail(row: dict[str, Any], *, include_config: bool) -> dict[str, Any]:
    config = _as_dict(row.get("config"))
    detail = {
        **_build_run_summary(row),
        "val_metrics": _json_safe(_as_dict(row.get("val_metrics"))),
        "test_metrics": _json_safe(_as_dict(row.get("test_metrics"))),
        "config_summary": _build_config_summary(config),
        "wandb_run_id": _first_value(row, "wandb_run_id"),
    }
    if include_config:
        detail["config"] = _json_safe(config)
    return detail


def _build_evaluation_summary(row: dict[str, Any]) -> dict[str, Any]:
    meta = _as_dict(row.get("meta"))
    return {
        "run_id": row["run_id"],
        "ticker": _first_value(row, "ticker", meta),
        "timeframe": _first_value(row, "timeframe", meta),
        "asof_date": _first_value(row, "asof_date", meta),
        "coverage": _first_value(row, "coverage", meta),
        "lower_breach_rate": _first_value(row, "lower_breach_rate", meta),
        "upper_breach_rate": _first_value(row, "upper_breach_rate", meta),
        "avg_band_width": _first_value(row, "avg_band_width", meta),
        "direction_accuracy": _first_value(row, "direction_accuracy", meta),
        "mae": _first_value(row, "mae", meta),
        "smape": _first_value(row, "smape", meta),
        "spearman_ic": _first_value(row, "spearman_ic", meta),
        "top_k_long_spread": _first_value(row, "top_k_long_spread", meta),
        "top_k_short_spread": _first_value(row, "top_k_short_spread", meta),
        "long_short_spread": _first_value(row, "long_short_spread", meta),
        "fee_adjusted_return": _first_value(row, "fee_adjusted_return", meta),
        "fee_adjusted_sharpe": _first_value(row, "fee_adjusted_sharpe", meta),
        "fee_adjusted_turnover": _first_value(row, "fee_adjusted_turnover", meta),
    }


def _build_backtest_summary(row: dict[str, Any]) -> dict[str, Any]:
    meta = _as_dict(row.get("meta"))
    fee_adjusted_return_pct = _first_value(row, "fee_adjusted_return_pct", meta)
    fee_adjusted_sharpe = _first_value(row, "fee_adjusted_sharpe", meta)
    return {
        "run_id": row["run_id"],
        "strategy_name": _first_value(row, "strategy_name", meta),
        "timeframe": _first_value(row, "timeframe", meta),
        "return_pct": _first_value(row, "return_pct", meta),
        "sharpe": _first_value(row, "sharpe", meta),
        "mdd": _first_value(row, "mdd", meta),
        "win_rate": _first_value(row, "win_rate", meta),
        "profit_factor": _first_value(row, "profit_factor", meta),
        "num_trades": _first_value(row, "num_trades", meta),
        "fee_adjusted_return_pct": (
            fee_adjusted_return_pct if fee_adjusted_return_pct is not None else _first_value(row, "return_pct", meta)
        ),
        "fee_adjusted_sharpe": fee_adjusted_sharpe if fee_adjusted_sharpe is not None else _first_value(row, "sharpe", meta),
        "avg_turnover": _first_value(row, "avg_turnover", meta),
        "meta": _json_safe(meta),
        "created_at": _first_value(row, "created_at", meta),
    }


@router.get(
    "/runs",
    response_model=ApiResponse[list[RunSummary]],
    responses={503: {"model": ErrorResponse}},
)
def list_ai_runs(
    request: Request,
    model_name: str | None = Query(default="patchtst", description="모델 이름"),
    timeframe: str | None = Query(default=None, description="타임프레임"),
    status: str | None = Query(default="completed", description="run 상태"),
    limit: int = Query(default=20, ge=1, le=100, description="반환할 최대 run 수"),
    offset: int = Query(default=0, ge=0, description="조회 시작 위치"),
):
    resolved_status = (status or "completed").strip() or "completed"
    rows = fetch_model_runs(
        model_name=model_name,
        timeframe=timeframe,
        status=resolved_status,
        limit=limit,
        offset=offset,
    )
    return success_response(request, [_build_run_summary(row) for row in rows], total=len(rows))


@router.get(
    "/runs/{run_id}",
    response_model=ApiResponse[RunDetail],
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_ai_run(
    request: Request,
    run_id: str,
    include_config: bool = Query(default=False, description="원본 config 포함 여부"),
):
    row = fetch_model_run(run_id)
    if row is None:
        raise ResourceNotFoundError(f"run_id={run_id} AI run을 찾을 수 없습니다.")
    return success_response(request, _build_run_detail(row, include_config=include_config))


@router.get(
    "/runs/{run_id}/evaluations",
    response_model=ApiResponse[list[EvaluationSummary]],
    responses={503: {"model": ErrorResponse}},
)
def list_run_evaluations(
    request: Request,
    run_id: str,
    ticker: str | None = Query(default=None, description="티커"),
    timeframe: str | None = Query(default=None, description="타임프레임"),
    limit: int = Query(default=100, ge=1, le=1000, description="반환할 최대 평가 수"),
):
    rows = fetch_run_evaluations(run_id, ticker=ticker, timeframe=timeframe, limit=limit)
    return success_response(request, [_build_evaluation_summary(row) for row in rows], total=len(rows))


@router.get(
    "/runs/{run_id}/backtests",
    response_model=ApiResponse[list[BacktestSummary]],
    responses={503: {"model": ErrorResponse}},
)
def list_run_backtests(
    request: Request,
    run_id: str,
    strategy_name: str | None = Query(default=None, description="전략 이름"),
    timeframe: str | None = Query(default=None, description="타임프레임"),
    limit: int = Query(default=50, ge=1, le=200, description="반환할 최대 백테스트 수"),
):
    rows = fetch_run_backtests(run_id, strategy_name=strategy_name, timeframe=timeframe, limit=limit)
    return success_response(request, [_build_backtest_summary(row) for row in rows], total=len(rows))
