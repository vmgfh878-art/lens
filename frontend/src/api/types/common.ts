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

export function isPredictionTimeframeEnabled(
  timeframe: DisplayTimeframe
): timeframe is PredictionTimeframe {
  return timeframe === "1D" || timeframe === "1W";
}
