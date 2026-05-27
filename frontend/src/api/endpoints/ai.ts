import { api } from "../baseClient";
import type { ApiResponse, DisplayTimeframe } from "../types/common";
import type {
  AiRunDetail,
  AiRunStatus,
  AiRunSummary,
  BacktestSummary,
  EvaluationSummary,
} from "../types/ai";

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
  const res = await api.get<ApiResponse<EvaluationSummary[]>>(
    `/api/v1/ai/runs/${runId}/evaluations`,
    { params }
  );
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
  const res = await api.get<ApiResponse<BacktestSummary[]>>(
    `/api/v1/ai/runs/${runId}/backtests`,
    { params }
  );
  return res.data;
}
