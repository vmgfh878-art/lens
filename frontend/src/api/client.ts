import axios from "axios";

// 우선순위: NEXT_PUBLIC_BACKEND_URL > 배포 기본 Render URL.
// 배포 화면에서는 localhost fallback 에 기대지 않도록 명시적 상태를 노출한다.
const PRODUCTION_BACKEND_URL = "https://lens-backend-7stj.onrender.com";
const LOCAL_BACKEND_URL = "http://127.0.0.1:8000";

type BackendUrlSource = "env" | "local-default" | "production-default" | "invalid-env";

interface BackendUrlResolution {
  url: string;
  source: BackendUrlSource;
  warning: string | null;
}

function isLocalBrowserHost() {
  if (typeof window === "undefined") {
    return false;
  }
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1";
}

function resolveBackendUrl(value: string | undefined): BackendUrlResolution {
  const raw = value?.trim().replace(/^["']|["']$/g, "");
  if (!raw) {
    if (isLocalBrowserHost()) {
      return {
        url: LOCAL_BACKEND_URL,
        source: "local-default",
        warning: "NEXT_PUBLIC_BACKEND_URL이 비어 있어 로컬 백엔드로 연결합니다. 배포 환경에서는 환경변수를 반드시 설정해야 합니다.",
      };
    }
    return {
      url: PRODUCTION_BACKEND_URL,
      source: "production-default",
      warning: "NEXT_PUBLIC_BACKEND_URL이 설정되지 않아 Render 기본 백엔드로 연결합니다. 배포 설정에서 환경변수를 확인해주세요.",
    };
  }

  const withProtocol = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`;
  try {
    const parsed = new URL(withProtocol);
    return {
      url: parsed.toString().replace(/\/$/, ""),
      source: "env",
      warning: null,
    };
  } catch {
    const fallback = isLocalBrowserHost() ? LOCAL_BACKEND_URL : PRODUCTION_BACKEND_URL;
    return {
      url: fallback,
      source: "invalid-env",
      warning: `NEXT_PUBLIC_BACKEND_URL 값이 올바르지 않아 ${fallback}로 연결합니다.`,
    };
  }
}

const backendUrlResolution = resolveBackendUrl(process.env.NEXT_PUBLIC_BACKEND_URL);

const api = axios.create({
  baseURL: backendUrlResolution.url,
});

export function getBackendBaseUrl() {
  return backendUrlResolution.url;
}

export function getBackendConfigWarning() {
  return backendUrlResolution.warning;
}

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

export interface IndicatorPoint {
  date: string;
  rsi: number | null;
  macd_ratio: number | null;
  bb_position: number | null;
  ma_5_ratio: number | null;
  ma_20_ratio: number | null;
  ma_60_ratio: number | null;
  vol_change: number | null;
  volume: number | null;
  atr_ratio: number | null;
  regime_label: string | null;
}

export interface IndicatorResult {
  ticker: string;
  timeframe: DisplayTimeframe;
  data: IndicatorPoint[];
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
  meta: Record<string, unknown>;
}

export interface ProductLineHistoryPoint {
  asof_date: string;
  display_horizon: number;
  value: number;
  run_id: string;
}

export interface ProductBandHistoryPoint {
  asof_date: string;
  display_horizon: number;
  lower: number;
  upper: number;
  run_id: string;
}

export interface ProductPredictionHistoryManifestSummary {
  line_run_id: string | null;
  band_run_id: string | null;
  date_range: {
    start?: string | null;
    end?: string | null;
  };
  row_count: number;
}

export interface ProductPredictionHistoryResult {
  ticker: string;
  timeframe: string;
  latest_asof_date: string | null;
  source: string;
  line_history: ProductLineHistoryPoint[];
  band_history: ProductBandHistoryPoint[];
  manifest_summary: ProductPredictionHistoryManifestSummary;
  empty_reason: string | null;
}

export interface V1LinePredictionPoint {
  ticker: string;
  asof_date: string;
  line_score: number | null;
  safe_line_score: number | null;
  line_rank_by_date?: number | null;
  safe_line_rank_by_date?: number | null;
  line_top_decile_flag?: boolean | number | null;
  safe_line_top_decile_flag?: boolean | number | null;
  actual_h5_return?: number | null;
  model_id?: string | null;
  source_cp?: string | null;
}

export interface V1BandPredictionPoint {
  ticker: string;
  asof_date: string;
  forecast_date?: string | null;
  horizon_step: number;
  band_lower: number | null;
  band_upper: number | null;
  actual_return?: number | null;
  actual_return_available?: boolean | null;
  model_id?: string | null;
  source_cp?: string | null;
}

export interface V1LinePredictionResult {
  ticker: string;
  slot: string;
  model_id: string | null;
  source_cp: string | null;
  rows: number;
  data: V1LinePredictionPoint[];
}

export interface V1BandPredictionResult {
  ticker: string;
  slot: string;
  model_id: string | null;
  source_cp: string | null;
  horizons: number[];
  rows: number;
  data: V1BandPredictionPoint[];
}

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

export async function fetchIndicators(
  ticker: string,
  options?: { timeframe?: DisplayTimeframe; limit?: number }
): Promise<ApiResponse<IndicatorResult>> {
  const params = {
    timeframe: options?.timeframe ?? "1D",
    limit: options?.limit ?? 300,
  };
  const res = await api.get<ApiResponse<IndicatorResult>>(`/api/v1/stocks/${ticker}/indicators`, { params });
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

export async function fetchPredictionHistory(
  ticker: string,
  options: {
    runId: string;
    limit?: number;
  }
): Promise<ApiResponse<PredictionResult[]>> {
  const params = {
    run_id: options.runId,
    limit: options.limit ?? 90,
  };
  const res = await api.get<ApiResponse<PredictionResult[]>>(`/api/v1/stocks/${ticker}/predictions/history`, { params });
  return res.data;
}

export async function fetchProductPredictionHistory(
  ticker: string,
  options?: {
    timeframe?: PredictionTimeframe;
    roles?: "all" | "line" | "band" | "line,band";
    runId?: string;
    limit?: number;
    lookbackDays?: number;
  }
): Promise<ApiResponse<ProductPredictionHistoryResult>> {
  const params = {
    timeframe: options?.timeframe ?? "1D",
    roles: options?.roles ?? "all",
    run_id: options?.runId,
    limit: options?.limit,
    lookback_days: options?.lookbackDays,
  };
  const res = await api.get<ApiResponse<ProductPredictionHistoryResult>>(
    `/api/v1/stocks/${ticker}/predictions/product-history`,
    { params }
  );
  return res.data;
}

export async function fetchV1LinePrediction(
  ticker: string,
  options?: { days?: number }
): Promise<ApiResponse<V1LinePredictionResult>> {
  const params = {
    days: options?.days ?? 365,
  };
  const res = await api.get<ApiResponse<V1LinePredictionResult>>(`/api/v1/predictions/line/${ticker}`, { params });
  return res.data;
}

export async function fetchV1Band1dPrediction(
  ticker: string,
  options?: { days?: number; horizon?: number }
): Promise<ApiResponse<V1BandPredictionResult>> {
  const params = {
    days: options?.days ?? 365,
    horizon: options?.horizon,
  };
  const res = await api.get<ApiResponse<V1BandPredictionResult>>(`/api/v1/predictions/band/1d/${ticker}`, { params });
  return res.data;
}

export async function fetchV1Band1wPrediction(
  ticker: string,
  options?: { days?: number; horizon?: number }
): Promise<ApiResponse<V1BandPredictionResult>> {
  const params = {
    days: options?.days ?? 730,
    horizon: options?.horizon,
  };
  const res = await api.get<ApiResponse<V1BandPredictionResult>>(`/api/v1/predictions/band/1w/${ticker}`, { params });
  return res.data;
}

export async function fetchAiRuns(options?: {
  modelName?: string;
  timeframe?: DisplayTimeframe;
  status?: AiRunStatus;
  includeLegacy?: boolean;
  limit?: number;
  offset?: number;
}): Promise<ApiResponse<AiRunSummary[]>> {
  const params = {
    model_name: options?.modelName ?? "patchtst",
    timeframe: options?.timeframe,
    status: options?.status ?? "completed",
    include_legacy: options?.includeLegacy ?? false,
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
