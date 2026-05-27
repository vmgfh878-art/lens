import { api } from "../baseClient";
import type { ApiResponse } from "../types/common";
import type { StrategyBacktestResult, StrategyScanResult } from "../types/strategies";

export async function fetchStrategyScan(
  strategyId: string,
  options?: { limit?: number }
): Promise<ApiResponse<StrategyScanResult>> {
  const res = await api.get<ApiResponse<StrategyScanResult>>(
    `/api/v1/strategies/${strategyId}/scan`,
    { params: { limit: options?.limit ?? 500 } }
  );
  return res.data;
}

export async function fetchStrategyBacktest(
  strategyId: string,
  ticker: string
): Promise<ApiResponse<StrategyBacktestResult>> {
  const res = await api.get<ApiResponse<StrategyBacktestResult>>(
    `/api/v1/strategies/${strategyId}/backtest/${ticker}`
  );
  return res.data;
}
