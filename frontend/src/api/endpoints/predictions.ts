import { api } from "../baseClient";
import type { ApiResponse, PredictionTimeframe } from "../types/common";
import type {
  PredictionResult,
  ProductPredictionHistoryResult,
  V1BandPredictionResult,
  V1LinePredictionResult,
} from "../types/predictions";

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
  const res = await api.get<ApiResponse<PredictionResult>>(
    `/api/v1/stocks/${ticker}/predictions/latest`,
    { params }
  );
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
  const res = await api.get<ApiResponse<PredictionResult[]>>(
    `/api/v1/stocks/${ticker}/predictions/history`,
    { params }
  );
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
  const res = await api.get<ApiResponse<V1LinePredictionResult>>(
    `/api/v1/predictions/line/${ticker}`,
    { params }
  );
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
  const res = await api.get<ApiResponse<V1BandPredictionResult>>(
    `/api/v1/predictions/band/1d/${ticker}`,
    { params }
  );
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
  const res = await api.get<ApiResponse<V1BandPredictionResult>>(
    `/api/v1/predictions/band/1w/${ticker}`,
    { params }
  );
  return res.data;
}
