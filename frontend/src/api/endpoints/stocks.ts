import { api } from "../baseClient";
import type { ApiResponse, DisplayTimeframe } from "../types/common";
import type { IndicatorResult } from "../types/indicators";
import type { PriceResult, StockSummary } from "../types/stocks";

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
