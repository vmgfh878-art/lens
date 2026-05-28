import { api } from "../baseClient";
import type { ApiResponse, PredictionTimeframe } from "../types/common";
import type {
  ProductPredictionHistoryResult,
  V1BandPredictionResult,
  V1LinePredictionResult,
} from "../types/predictions";

// legacy fetchPrediction / fetchPredictionHistory 는 backend 의
// /api/v1/stocks/{ticker}/predictions/latest|history endpoint 와 함께 제거됐다.
// 모든 line/band 조회는 v1 endpoint (/api/v1/predictions/...) 로 단일화.

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
