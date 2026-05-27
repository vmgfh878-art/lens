export type StrategyId =
  | "indicator_balance_v2"
  | "ai_balance_v2"
  | "ai_band_defense_v1"
  | "indicator_baseline_v1"
  | "lens_balance_v1";

export type SignalGroupId = "buy" | "hold" | "risk" | "watch";

export type DecisionFactorId =
  | "conservative"
  | "lowerBand"
  | "bandWidth"
  | "bandExpansion"
  | "bandPercentile"
  | "ma60Trend"
  | "ma20Trend"
  | "macd"
  | "rsi"
  | "atr";

export interface StrategyDefinition {
  id: StrategyId;
  label: string;
  shortLabel: string;
  description: string;
  ruleRows: Array<[string, string]>;
  visibleFactors: DecisionFactorId[];
  validationRows: ReadonlyArray<readonly [string, string]>;
  scopeNote: string;
  usesAi: boolean;
}

export interface RiskSignal {
  date: string;
  position: 0 | 1;
  targetPosition: 0 | 1;
  conservativeReturn: number | null;
  lowerBandReturn: number | null;
  bandWidthReturn: number | null;
  bandWidthExpansion: number | null;
  bandWidthPercentile: number | null;
  ma60Ratio: number | null;
  ma20Ratio: number | null;
  macdRatio: number | null;
  rsi: number | null;
  atrRatio: number | null;
  reason: string;
}

export interface BacktestPoint {
  date: string;
  price: number;
  strategyEquity: number;
  buyHoldEquity: number;
  position: 0 | 1;
}

export interface TradeEvent {
  date: string;
  kind: "entry" | "exit";
  price: number;
  reason: string;
}

export interface BacktestSimulationResult {
  points: BacktestPoint[];
  signals: RiskSignal[];
  tradeEvents: TradeEvent[];
  strategyReturnPct: number;
  buyHoldReturnPct: number;
  buyHoldReturnRatio: number | null;
  excessReturnPct: number;
  maxDrawdownPct: number;
  buyHoldMaxDrawdownPct: number;
  maxDrawdownImprovementPct: number;
  feeAdjustedReturnPct: number;
  feeAdjustedSharpe: number;
  buyHoldSharpe: number;
  strategySortino: number;
  buyHoldSortino: number;
  tradeCount: number;
  cashWaitRatio: number;
  marketParticipationRate: number;
  worstTradeLossPct: number | null;
  averageHoldingDays: number | null;
  avoidedLargeLossDays: number;
  largeLossDays: number;
  largeLossAvoidanceRate: number | null;
  largeLossThresholdPct: number | null;
}

export interface LineSeries {
  label: string;
  color: string;
  values: Array<{ date: string; value: number }>;
}

export interface TradeRecord {
  date: string;
  action: string;
  price: number | null;
  reason: string;
  nextDayReturn: number | null;
  conservativeReturn: number | null;
  lowerBandReturn: number | null;
  bandWidthReturn: number | null;
  bandWidthExpansion: number | null;
  bandWidthPercentile: number | null;
  ma60Ratio: number | null;
  ma20Ratio: number | null;
  macdRatio: number | null;
  rsi: number | null;
  atrRatio: number | null;
}

export interface StrategySignalCard {
  ticker: string;
  sector: string | null;
  group: SignalGroupId;
  signalLabel: string;
  reason: string;
  asofDate: string | null;
  conservativeReturn: number | null;
  lowerBandReturn: number | null;
  bandWidthReturn: number | null;
  bandWidthExpansion: number | null;
  bandWidthPercentile: number | null;
  ma60Ratio: number | null;
  ma20Ratio: number | null;
  macdRatio: number | null;
  rsi: number | null;
  atrRatio: number | null;
  strategyScore?: number | null;
  hasUsableSignal: boolean;
}
