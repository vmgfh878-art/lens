import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000",
});

export type DisplayTimeframe = "1D" | "1W" | "1M";
export type PredictionTimeframe = "1D" | "1W";

export interface ApiMeta {
  request_id: string;
  [key: string]: unknown;
}

export interface ApiResponse<T> {
  data: T;
  meta: ApiMeta;
}

export interface PriceBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

export interface StockSummary {
  ticker: string;
  sector: string | null;
  industry: string | null;
  market_cap: number | null;
}

export interface PriceResult {
  ticker: string;
  timeframe: DisplayTimeframe;
  start: string;
  end: string;
  data: PriceBar[];
}

export interface PredictionResult {
  ticker: string;
  model_name: string;
  timeframe: PredictionTimeframe;
  horizon: number;
  asof_date: string;
  decision_time: string;
  run_id: string;
  model_ver: string;
  signal: "BUY" | "SELL" | "HOLD";
  forecast_dates: string[];
  upper_band_series: number[];
  lower_band_series: number[];
  conservative_series: number[];
  line_series: number[];
  band_quantile_low: number | null;
  band_quantile_high: number | null;
}

export type AiRunStatus = "completed" | "failed_nan";

export interface AiRunSummary {
  run_id: string;
  status: AiRunStatus | string | null;
  model_name: string | null;
  timeframe: DisplayTimeframe | string | null;
  horizon: number | null;
  created_at: string | null;
  model_ver: string | null;
  checkpoint_path: string | null;
  best_epoch: number | null;
  best_val_total: number | null;
  line_target_type: string | null;
  band_target_type: string | null;
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

export function isPredictionTimeframeEnabled(timeframe: DisplayTimeframe): timeframe is PredictionTimeframe {
  return timeframe === "1D" || timeframe === "1W";
}

export async function fetchPrices(
  ticker: string,
  options?: { start?: string; end?: string; timeframe?: DisplayTimeframe }
): Promise<ApiResponse<PriceResult>> {
  const params = {
    start: options?.start,
    end: options?.end,
    timeframe: options?.timeframe ?? "1D",
  };
  const res = await api.get<ApiResponse<PriceResult>>(`/api/v1/stocks/${ticker}/prices`, { params });
  return res.data;
}

export async function fetchTickers(options?: {
  search?: string;
  limit?: number;
}): Promise<ApiResponse<StockSummary[]>> {
  const params = {
    search: options?.search,
    limit: options?.limit,
  };
  const res = await api.get<ApiResponse<StockSummary[]>>("/api/v1/stocks", { params });
  return res.data;
}

export async function fetchPrediction(
  ticker: string,
  options?: {
    model?: "patchtst" | "cnn_lstm" | "tide";
    timeframe?: PredictionTimeframe;
    horizon?: number;
    runId?: string;
  }
): Promise<ApiResponse<PredictionResult>> {
  const params = {
    model: options?.model ?? "patchtst",
    timeframe: options?.timeframe ?? "1D",
    horizon: options?.horizon,
    run_id: options?.runId,
  };
  const res = await api.get<ApiResponse<PredictionResult>>(`/api/v1/stocks/${ticker}/predictions/latest`, { params });
  return res.data;
}

export async function fetchAiRuns(options?: {
  modelName?: string;
  timeframe?: DisplayTimeframe;
  status?: AiRunStatus;
  limit?: number;
  offset?: number;
}): Promise<ApiResponse<AiRunSummary[]>> {
  const params = {
    model_name: options?.modelName ?? "patchtst",
    timeframe: options?.timeframe,
    status: options?.status ?? "completed",
    limit: options?.limit ?? 20,
    offset: options?.offset ?? 0,
  };
  const res = await api.get<ApiResponse<AiRunSummary[]>>("/api/v1/ai/runs", { params });
  return res.data;
}

export async function fetchAiRun(
  runId: string,
  options?: { includeConfig?: boolean }
): Promise<ApiResponse<AiRunDetail>> {
  const res = await api.get<ApiResponse<AiRunDetail>>(`/api/v1/ai/runs/${runId}`, {
    params: { include_config: options?.includeConfig ?? false },
  });
  return res.data;
}

export async function fetchRunEvaluations(
  runId: string,
  options?: {
    ticker?: string;
    timeframe?: DisplayTimeframe;
    limit?: number;
  }
): Promise<ApiResponse<EvaluationSummary[]>> {
  const params = {
    ticker: options?.ticker,
    timeframe: options?.timeframe,
    limit: options?.limit ?? 100,
  };
  const res = await api.get<ApiResponse<EvaluationSummary[]>>(`/api/v1/ai/runs/${runId}/evaluations`, { params });
  return res.data;
}

export async function fetchRunBacktests(
  runId: string,
  options?: {
    strategyName?: string;
    timeframe?: DisplayTimeframe;
    limit?: number;
  }
): Promise<ApiResponse<BacktestSummary[]>> {
  const params = {
    strategy_name: options?.strategyName,
    timeframe: options?.timeframe,
    limit: options?.limit ?? 50,
  };
  const res = await api.get<ApiResponse<BacktestSummary[]>>(`/api/v1/ai/runs/${runId}/backtests`, { params });
  return res.data;
}
