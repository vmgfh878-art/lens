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
  }
): Promise<ApiResponse<PredictionResult>> {
  const params = {
    model: options?.model ?? "patchtst",
    timeframe: options?.timeframe ?? "1D",
    horizon: options?.horizon,
  };
  const res = await api.get<ApiResponse<PredictionResult>>(`/api/v1/stocks/${ticker}/predictions/latest`, { params });
  return res.data;
}
