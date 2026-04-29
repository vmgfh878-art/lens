from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunSummary(BaseModel):
    run_id: str
    status: str | None = None
    model_name: str | None = None
    timeframe: str | None = None
    horizon: int | None = None
    created_at: str | None = None
    model_ver: str | None = None
    feature_version: str | None = None
    band_mode: str | None = None
    checkpoint_path: str | None = None
    best_epoch: int | None = None
    best_val_total: float | None = None
    line_target_type: str | None = None
    band_target_type: str | None = None


class RunDetail(RunSummary):
    val_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    config_summary: dict[str, Any] = Field(default_factory=dict)
    wandb_run_id: str | None = None
    config: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class EvaluationSummary(BaseModel):
    run_id: str
    ticker: str | None = None
    timeframe: str | None = None
    asof_date: str | None = None
    coverage: float | None = None
    lower_breach_rate: float | None = None
    upper_breach_rate: float | None = None
    avg_band_width: float | None = None
    direction_accuracy: float | None = None
    mae: float | None = None
    smape: float | None = None
    spearman_ic: float | None = None
    top_k_long_spread: float | None = None
    top_k_short_spread: float | None = None
    long_short_spread: float | None = None
    fee_adjusted_return: float | None = None
    fee_adjusted_sharpe: float | None = None
    fee_adjusted_turnover: float | None = None


class BacktestSummary(BaseModel):
    run_id: str
    strategy_name: str | None = None
    timeframe: str | None = None
    return_pct: float | None = None
    sharpe: float | None = None
    mdd: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    num_trades: int | None = None
    fee_adjusted_return_pct: float | None = None
    fee_adjusted_sharpe: float | None = None
    avg_turnover: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
