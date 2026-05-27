// Backtest 시뮬레이션 결과를 사용자 친화적 라벨로 변환.

import type { BacktestSimulationResult } from "./types";

export function describeAvoidanceStrength(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  if (value >= 0.6) {
    return "강함";
  }
  if (value >= 0.4) {
    return "보통";
  }
  return "약함";
}

export function evaluateFollowReturn(result: BacktestSimulationResult | null) {
  if (!result || result.buyHoldReturnRatio == null) {
    return "-";
  }
  if (result.buyHoldReturnPct < 0 && result.strategyReturnPct > result.buyHoldReturnPct) {
    return "방어 우위";
  }
  if (result.buyHoldReturnRatio >= 0.7) {
    return "양호";
  }
  if (result.buyHoldReturnRatio >= 0.4) {
    return "보통";
  }
  return "약함";
}

export function evaluateDrawdown(result: BacktestSimulationResult | null) {
  if (!result) {
    return "-";
  }
  if (result.maxDrawdownImprovementPct >= 5) {
    return "양호";
  }
  if (result.maxDrawdownImprovementPct > 0) {
    return "보통";
  }
  return "약함";
}

export function evaluateLossAvoidance(result: BacktestSimulationResult | null) {
  return describeAvoidanceStrength(result?.largeLossAvoidanceRate);
}

export function evaluateTradeFrequency(result: BacktestSimulationResult | null) {
  if (!result) {
    return "-";
  }
  if (result.tradeCount <= 20) {
    return "적정";
  }
  if (result.tradeCount <= 40) {
    return "많음";
  }
  return "과도";
}
