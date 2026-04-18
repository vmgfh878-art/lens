import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000",
});

export interface PriceBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionResult {
  ticker: string;
  model_name: string;
  timeframe: "1D" | "1W" | "1M";
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
}

export async function fetchPrices(
  ticker: string,
  options?: { start?: string; end?: string; timeframe?: "1D" | "1W" | "1M" }
): Promise<PriceBar[]> {
  const params = {
    start: options?.start,
    end: options?.end,
    timeframe: options?.timeframe ?? "1D",
  };
  const res = await api.get(`/prices/${ticker}`, { params });
  return res.data.data;
}

export async function fetchTickers() {
  const res = await api.get("/prices/");
  return res.data.tickers;
}

export async function fetchPrediction(
  ticker: string,
  options?: {
    model?: "patchtst" | "cnn_lstm" | "tide";
    timeframe?: "1D" | "1W" | "1M";
    horizon?: number;
  }
): Promise<PredictionResult> {
  const params = {
    model: options?.model ?? "patchtst",
    timeframe: options?.timeframe ?? "1D",
    horizon: options?.horizon,
  };
  const res = await api.get(`/predict/${ticker}`, { params });
  return res.data;
}
