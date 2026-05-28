// Backward-compatible re-export.
// 기존 코드: import { fetchPrices, PriceBar, ... } from "@/api/client"
// 새 구조:   src/api/{baseClient, types/*, endpoints/*}
// 점진적으로 컴포넌트가 endpoint 별 import 로 옮겨가도 되고, 그대로 둬도 됨.

export { api, getBackendBaseUrl, getBackendConfigWarning } from "./baseClient";

export type {
  ApiMeta,
  ApiResponse,
  DisplayTimeframe,
  PredictionTimeframe,
} from "./types/common";
export { isPredictionTimeframeEnabled } from "./types/common";

export type { PriceBar, PriceResult, StockSummary } from "./types/stocks";
export type { IndicatorPoint, IndicatorResult } from "./types/indicators";
export type {
  PredictionResult,
  ProductBandHistoryPoint,
  ProductLineHistoryPoint,
  ProductPredictionHistoryManifestSummary,
  ProductPredictionHistoryResult,
  V1BandPredictionPoint,
  V1BandPredictionResult,
  V1LinePredictionPoint,
  V1LinePredictionResult,
} from "./types/predictions";
export type {
  AiRunDetail,
  AiRunStatus,
  AiRunSummary,
  BacktestSummary,
  EvaluationSummary,
} from "./types/ai";
export type {
  StrategyBacktestResult,
  StrategyPortfolioMetrics,
  StrategyScanResult,
} from "./types/strategies";

export {
  fetchIndicators,
  fetchPrices,
  fetchTickers,
} from "./endpoints/stocks";
export {
  fetchProductPredictionHistory,
  fetchV1Band1dPrediction,
  fetchV1Band1wPrediction,
  fetchV1LinePrediction,
} from "./endpoints/predictions";
export {
  fetchAiRun,
  fetchAiRuns,
  fetchRunBacktests,
  fetchRunEvaluations,
} from "./endpoints/ai";
export {
  fetchStrategyBacktest,
  fetchStrategyScan,
} from "./endpoints/strategies";
