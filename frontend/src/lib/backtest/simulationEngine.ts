// 백테스트 시뮬레이션 + 통계 순수 함수.
// signalBuilder가 만든 신호를 가격에 적용해 BacktestSimulationResult를 만든다.

import type { IndicatorPoint, PredictionResult, PriceBar } from "@/api/client";
import { buildIndicatorSignals, buildSignals } from "@/lib/backtest/signalBuilder";
import type {
  BacktestPoint,
  BacktestSimulationResult,
  StrategyId,
  TradeEvent,
} from "@/lib/backtest/types";

export function calculateMaxDrawdown(points: BacktestPoint[], equityKey: "strategyEquity" | "buyHoldEquity" = "strategyEquity") {
  let peak = 1;
  let maxDrawdown = 0;
  points.forEach((point) => {
    peak = Math.max(peak, point[equityKey]);
    if (peak > 0) {
      maxDrawdown = Math.min(maxDrawdown, point[equityKey] / peak - 1);
    }
  });
  return maxDrawdown * 100;
}

export function calculateSharpe(returns: number[]) {
  if (returns.length < 2) {
    return 0;
  }
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (returns.length - 1);
  const std = Math.sqrt(variance);
  return std > 0 ? (mean / std) * Math.sqrt(252) : 0;
}

export function calculateSortino(returns: number[]) {
  if (returns.length < 2) {
    return 0;
  }
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const downside = returns.filter((value) => value < 0);
  if (downside.length < 2) {
    return mean > 0 ? mean * Math.sqrt(252) : 0;
  }
  const downsideVariance = downside.reduce((sum, value) => sum + value ** 2, 0) / (downside.length - 1);
  const downsideDeviation = Math.sqrt(downsideVariance);
  return downsideDeviation > 0 ? (mean / downsideDeviation) * Math.sqrt(252) : 0;
}

export function quantile(values: number[], ratio: number) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * ratio)));
  return sorted[index];
}

export function chooseLargeLossThreshold(returns: number[]) {
  const lowerQuintile = quantile(returns, 0.2);
  if (lowerQuintile == null) {
    return null;
  }
  return Math.min(-0.02, lowerQuintile);
}

export function runStrategyBacktest(params: {
  strategyId: StrategyId;
  priceRows: PriceBar[];
  lineHistory: PredictionResult[];
  bandHistory: PredictionResult[];
  indicators: IndicatorPoint[];
  feeBps: number;
}): BacktestSimulationResult | null {
  const signals =
    params.strategyId === "lens_balance_v1"
      ? buildSignals(params)
      : buildIndicatorSignals({ priceRows: params.priceRows, indicators: params.indicators });
  if (params.priceRows.length < 2 || signals.length === 0) {
    return null;
  }

  const firstSignalDate = signals[0].date;
  const signalByDate = new Map(signals.map((signal) => [signal.date, signal]));
  const priceRows = params.priceRows.filter((row) => row.date >= firstSignalDate);
  if (priceRows.length < 2) {
    return null;
  }

  const feeRate = params.feeBps / 10000;
  const buyHoldDailyReturns = priceRows.slice(1).map((row, index) => row.close / priceRows[index].close - 1);
  const largeLossThreshold = chooseLargeLossThreshold(buyHoldDailyReturns);
  const points: BacktestPoint[] = [
    {
      date: priceRows[0].date,
      price: priceRows[0].close,
      strategyEquity: 1,
      buyHoldEquity: 1,
      position: 0,
    },
  ];
  const tradeEvents: TradeEvent[] = [];
  const dailyStrategyReturns: number[] = [];
  const tradeReturns: number[] = [];
  const holdingDurations: number[] = [];

  let strategyEquity = 1;
  let buyHoldEquity = 1;
  let position: 0 | 1 = 0;
  let tradeCount = 0;
  let cashDays = 0;
  let avoidedLargeLossDays = 0;
  let largeLossDays = 0;
  let entryPrice: number | null = null;
  let entryIndex: number | null = null;

  for (let index = 1; index < priceRows.length; index += 1) {
    const previous = priceRows[index - 1];
    const current = priceRows[index];
    const signal = signalByDate.get(previous.date);
    let desiredPosition: 0 | 1 = position;
    let tradeReason = signal?.reason ?? "직전 신호를 유지합니다.";
    let feeCost = 0;

    if (signal) {
      desiredPosition = signal.position;
    }

    if (desiredPosition !== position) {
      strategyEquity *= 1 - feeRate;
      feeCost = feeRate;
      tradeCount += 1;
      if (desiredPosition === 1) {
        entryPrice = previous.close;
        entryIndex = index - 1;
        tradeEvents.push({ date: previous.date, kind: "entry", price: previous.close, reason: tradeReason });
      } else {
        tradeEvents.push({ date: previous.date, kind: "exit", price: previous.close, reason: tradeReason });
        if (entryPrice && entryIndex != null) {
          tradeReturns.push(previous.close / entryPrice - 1);
          holdingDurations.push(index - 1 - entryIndex);
        }
        entryPrice = null;
        entryIndex = null;
      }
      position = desiredPosition;
    }

    const dailyReturn = current.close / previous.close - 1;
    const strategyDailyReturn = position === 1 ? dailyReturn : 0;
    strategyEquity *= 1 + strategyDailyReturn;
    buyHoldEquity *= 1 + dailyReturn;
    dailyStrategyReturns.push(strategyDailyReturn - feeCost);

    if (largeLossThreshold != null && dailyReturn <= largeLossThreshold) {
      largeLossDays += 1;
      if (position === 0) {
        avoidedLargeLossDays += 1;
      }
    }
    if (position === 0) {
      cashDays += 1;
    }

    points.push({
      date: current.date,
      price: current.close,
      strategyEquity,
      buyHoldEquity,
      position,
    });
  }

  if (position === 1 && entryPrice && entryIndex != null) {
    const last = priceRows[priceRows.length - 1];
    tradeReturns.push(last.close / entryPrice - 1);
    holdingDurations.push(priceRows.length - 1 - entryIndex);
  }

  const strategyReturnPct = (strategyEquity - 1) * 100;
  const buyHoldReturnPct = (buyHoldEquity - 1) * 100;
  const maxDrawdownPct = calculateMaxDrawdown(points);
  const buyHoldMaxDrawdownPct = calculateMaxDrawdown(points, "buyHoldEquity");
  const averageHoldingDays =
    holdingDurations.length > 0 ? holdingDurations.reduce((sum, value) => sum + value, 0) / holdingDurations.length : null;

  return {
    points,
    signals,
    tradeEvents,
    strategyReturnPct,
    buyHoldReturnPct,
    buyHoldReturnRatio: buyHoldReturnPct !== 0 ? strategyReturnPct / buyHoldReturnPct : null,
    excessReturnPct: strategyReturnPct - buyHoldReturnPct,
    maxDrawdownPct,
    buyHoldMaxDrawdownPct,
    maxDrawdownImprovementPct: maxDrawdownPct - buyHoldMaxDrawdownPct,
    feeAdjustedReturnPct: strategyReturnPct,
    feeAdjustedSharpe: calculateSharpe(dailyStrategyReturns),
    buyHoldSharpe: calculateSharpe(buyHoldDailyReturns),
    strategySortino: calculateSortino(dailyStrategyReturns),
    buyHoldSortino: calculateSortino(buyHoldDailyReturns),
    tradeCount,
    cashWaitRatio: points.length > 1 ? cashDays / (points.length - 1) : 0,
    marketParticipationRate: points.length > 1 ? 1 - cashDays / (points.length - 1) : 0,
    worstTradeLossPct: tradeReturns.length > 0 ? Math.min(...tradeReturns) * 100 : null,
    averageHoldingDays,
    avoidedLargeLossDays,
    largeLossDays,
    largeLossAvoidanceRate: largeLossDays > 0 ? avoidedLargeLossDays / largeLossDays : null,
    largeLossThresholdPct: largeLossThreshold == null ? null : largeLossThreshold * 100,
  };
}
