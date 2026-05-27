import type { PredictionTimeframe } from "./common";

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
  forecast_date?: string | null;
  display_horizon: number;
  value: number;
  run_id: string;
}

export interface ProductBandHistoryPoint {
  asof_date: string;
  forecast_date?: string | null;
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
