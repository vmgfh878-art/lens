import type { BacktestSimulationResult, StrategySignalCard } from "@/lib/backtest/types";

export interface StrategyPortfolioMetrics {
  strategyReturnPct: number;
  buyHoldReturnPct: number;
  excessReturnPct: number;
  maxDrawdownPct: number;
  buyHoldMaxDrawdownPct: number;
  maxDrawdownImprovementPct: number;
  feeAdjustedSharpe: number;
  buyHoldSharpe: number;
  strategySortino: number;
  buyHoldSortino: number;
  marketParticipationRate: number;
  avgSelectedCount: number | null;
  avgTradeCount?: number;
  largeLossAvoidanceRate?: number;
  passTickerRatio?: number;
}

export interface StrategyScanResult {
  strategyId: string;
  strategyLabel: string;
  timeframe: "1D";
  asofDate: string;
  scopeTickerCount: number;
  usableSignalCount: number;
  latestValidTickerCount?: number;
  cards: StrategySignalCard[];
  portfolioMetrics: StrategyPortfolioMetrics;
  aggregateMetrics?: StrategyPortfolioMetrics;
  contract?: string;
}

export type StrategyBacktestResult = BacktestSimulationResult;
