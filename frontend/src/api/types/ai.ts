import type { DisplayTimeframe } from "./common";

export type AiRunStatus = "completed" | "failed_nan" | "failed_quality_gate";

export interface AiRunSummary {
  run_id: string;
  status: AiRunStatus | string | null;
  model_name: string | null;
  timeframe: DisplayTimeframe | string | null;
  horizon: number | null;
  created_at: string | null;
  model_ver: string | null;
  feature_version: string | null;
  band_mode: string | null;
  checkpoint_path: string | null;
  best_epoch: number | null;
  best_val_total: number | null;
  line_target_type: string | null;
  band_target_type: string | null;
  role: string | null;
  feature_set: string | null;
  checkpoint_selection: string | null;
  wandb_status: string | null;
  deprecated_for_phase1_product_contract: boolean | string | null;
  indicator_layer_replacement: string | null;
  is_legacy: boolean;
}

export interface AiRunDetail extends AiRunSummary {
  val_metrics: Record<string, unknown>;
  test_metrics: Record<string, unknown>;
  config_summary: Record<string, unknown>;
  wandb_run_id: string | null;
  config?: Record<string, unknown> | null;
}

export interface EvaluationSummary {
  run_id: string;
  ticker: string | null;
  timeframe: string | null;
  asof_date: string | null;
  coverage: number | null;
  avg_band_width: number | null;
  direction_accuracy: number | null;
  mae: number | null;
  smape: number | null;
  spearman_ic: number | null;
  top_k_long_spread: number | null;
  top_k_short_spread: number | null;
  long_short_spread: number | null;
  fee_adjusted_return: number | null;
  fee_adjusted_sharpe: number | null;
  fee_adjusted_turnover: number | null;
}

export interface BacktestSummary {
  run_id: string;
  strategy_name: string | null;
  timeframe: string | null;
  return_pct: number | null;
  sharpe: number | null;
  mdd: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  num_trades: number | null;
  fee_adjusted_return_pct: number | null;
  fee_adjusted_sharpe: number | null;
  avg_turnover: number | null;
  meta: Record<string, unknown>;
  created_at: string | null;
}
