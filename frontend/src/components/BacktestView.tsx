"use client";

import { FormEvent, Fragment, startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import {
  DisplayTimeframe,
  fetchIndicators,
  fetchPrices,
  fetchStrategyBacktest,
  fetchStrategyScan,
  fetchTickers,
  IndicatorPoint,
  PredictionResult,
  PriceBar,
  StockSummary,
} from "@/api/client";
import MetricCard from "@/components/MetricCard";
import {
  DEFAULT_FEE_BPS,
  DEFAULT_PRICE_LOOKBACK_DAYS,
  getStrategyDefinition,
  INDICATOR_BASELINE_RULE,
  LENS_BALANCE_RULE,
  PREDICTION_HISTORY_LIMIT,
  PRODUCT_BAND_RUN_ID,
  PRODUCT_LINE_RUN_ID,
  SIGNAL_GROUP_DEFAULT_LIMIT,
  SIGNAL_GROUP_MAX_LIMIT,
  SIGNAL_GROUPS,
  SIGNAL_SCAN_TICKERS,
  STRATEGIES,
  isBackendStrategy,
  strategyNeedsBand,
  strategyNeedsLine,
} from "@/lib/backtest/constants";
import {
  describeAvoidanceStrength,
  evaluateDrawdown,
  evaluateFollowReturn,
  evaluateLossAvoidance,
  evaluateTradeFrequency,
} from "@/lib/backtest/evaluators";
import {
  getBandWidthValue,
  getConservativeValue,
  getHighestUpperBandValue,
  getWorstLowerBandValue,
  median,
  percentileRank,
} from "@/lib/backtest/predictionHelpers";
import type {
  BacktestPoint,
  BacktestSimulationResult,
  DecisionFactorId,
  LineSeries,
  RiskSignal,
  SignalGroupId,
  StrategyDefinition,
  StrategyId,
  StrategySignalCard,
  TradeEvent,
  TradeRecord,
} from "@/lib/backtest/types";
import {
  buildDefaultPriceWindow,
  isValidDate,
  sortPriceRows,
  sortUniqueByDate,
} from "@/lib/dateUtils";
import {
  formatCompact,
  formatNumber,
  formatPercent,
  formatRatioAsPercent,
  formatUnsignedRatioAsPercent,
  isFiniteNumber,
} from "@/lib/formatters";
import { PRODUCT_SLOT_BY_ID } from "@/lib/productSlots";

// BacktestView 전용 에러 메시지 (StockView 와 별도 톤).
function extractErrorMessage(error: unknown) {
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 설정과 백엔드 상태를 확인해주세요.";
    }
    return "백테스트 데이터를 불러오는 중 문제가 생겼습니다.";
  }
  return "백테스트 데이터를 불러오지 못했습니다.";
}

async function fetchPriceHistory(ticker: string, timeframe: DisplayTimeframe) {
  const window = buildDefaultPriceWindow(DEFAULT_PRICE_LOOKBACK_DAYS);
  const response = await fetchPrices(ticker, {
    timeframe,
    start: window.start,
    end: window.end,
  });
  return sortPriceRows(response.data.data);
}

function buildRawSignals(params: {
  priceRows: PriceBar[];
  lineHistory: PredictionResult[];
  bandHistory: PredictionResult[];
}) {
  const { priceRows, lineHistory, bandHistory } = params;
  const priceByDate = new Map(priceRows.map((row) => [row.date, row]));
  const lineByDate = new Map(lineHistory.map((row) => [row.asof_date, row]));
  const bandByDate = new Map(bandHistory.map((row) => [row.asof_date, row]));
  const candidateDates = Array.from(new Set([...lineHistory.map((row) => row.asof_date), ...bandHistory.map((row) => row.asof_date)])).sort(
    (left, right) => left.localeCompare(right)
  );

  const rows = candidateDates
    .map((date) => {
      const linePrediction = lineByDate.get(date);
      const bandPrediction = bandByDate.get(date);
      const price = priceByDate.get(date);
      if (!price || !linePrediction || !bandPrediction) {
        return null;
      }
      const conservativeValue = getConservativeValue(linePrediction);
      const lowerBandValue = getWorstLowerBandValue(bandPrediction);
      const bandWidthValue = getBandWidthValue(bandPrediction);
      if (!isFiniteNumber(conservativeValue) || !isFiniteNumber(lowerBandValue) || !isFiniteNumber(bandWidthValue)) {
        return null;
      }
      return {
        date,
        conservativeReturn: conservativeValue / price.close - 1,
        lowerBandReturn: lowerBandValue / price.close - 1,
        bandWidthReturn: bandWidthValue / price.close,
        ma60Ratio: null,
        ma20Ratio: null,
        macdRatio: null,
        rsi: null,
        atrRatio: null,
      };
    })
    .filter(
      (
        row
      ): row is {
        date: string;
        conservativeReturn: number;
        lowerBandReturn: number;
        bandWidthReturn: number;
        ma60Ratio: null;
        ma20Ratio: null;
        macdRatio: null;
        rsi: null;
        atrRatio: null;
      } => row !== null
    )
    .sort((left, right) => left.date.localeCompare(right.date));

  const allWidths = rows.map((row) => row.bandWidthReturn).filter(Number.isFinite);
  return rows.map((row, index) => {
    const previousWidths = rows
      .slice(Math.max(0, index - 20), index)
      .map((candidate) => candidate.bandWidthReturn)
      .filter(Number.isFinite);
    const widthReference = previousWidths.length >= 5 ? median(previousWidths) : null;
    return {
      ...row,
      bandWidthExpansion: widthReference && widthReference > 0 ? row.bandWidthReturn / widthReference : null,
      bandWidthPercentile: percentileRank(row.bandWidthReturn, allWidths),
    };
  });
}

function getRawTarget(row: {
  conservativeReturn: number | null;
  lowerBandReturn: number | null;
  bandWidthExpansion: number | null;
}) {
  const conservativeReturn = row.conservativeReturn ?? -Infinity;
  const lowerBandReturn = row.lowerBandReturn ?? Infinity;
  const lineGood = conservativeReturn >= LENS_BALANCE_RULE.lineEntryThreshold;
  const lineHold = conservativeReturn >= LENS_BALANCE_RULE.lineHoldThreshold;
  const lowerRisky = lowerBandReturn <= LENS_BALANCE_RULE.lowerRiskThreshold;
  const widthExpanded =
    row.bandWidthExpansion != null && row.bandWidthExpansion >= LENS_BALANCE_RULE.widthExpansionThreshold;
  const entryOk = lineGood && !widthExpanded;
  const riskExit = conservativeReturn < LENS_BALANCE_RULE.lineHoldThreshold && (lowerRisky || widthExpanded);

  if (entryOk) {
    return { target: 1 as const, reason: "보수적 기준선이 진입 기준을 충족하고 밴드 폭 급확장이 없습니다." };
  }
  if (lineHold && !riskExit) {
    return { target: 1 as const, reason: "보수적 기준선이 보유 기준 위이고 밴드 위험이 확정되지 않았습니다." };
  }
  if (riskExit && lowerRisky && widthExpanded) {
    return { target: 0 as const, reason: "예측선 약화와 밴드 하단 위험, 밴드 폭 확장이 동시에 나타났습니다." };
  }
  if (riskExit && lowerRisky) {
    return { target: 0 as const, reason: "예측선 약화와 밴드 하단 위험이 함께 나타났습니다." };
  }
  if (riskExit && widthExpanded) {
    return { target: 0 as const, reason: "예측선 약화와 밴드 폭 확장이 함께 나타났습니다." };
  }
  return { target: 0 as const, reason: "진입 또는 보유 기준을 충족하지 못해 현금으로 대기합니다." };
}

function normalizeRsi(value: number | null | undefined) {
  if (!Number.isFinite(value ?? NaN)) {
    return null;
  }
  const number = Number(value);
  return number >= 0 && number <= 1 ? number * 100 : number;
}

function buildIndicatorRows(priceRows: PriceBar[], indicators: IndicatorPoint[]) {
  const indicatorByDate = new Map(indicators.map((row) => [row.date, row]));
  const ma20Values = priceRows.map((row, index) => {
    const window = priceRows.slice(Math.max(0, index - 19), index + 1);
    if (window.length < 15) {
      return null;
    }
    const average = window.reduce((sum, item) => sum + item.close, 0) / window.length;
    return average > 0 ? priceRows[index].close / average - 1 : null;
  });
  const ma60Values = priceRows.map((row, index) => {
    const window = priceRows.slice(Math.max(0, index - 59), index + 1);
    if (window.length < 40) {
      return null;
    }
    const average = window.reduce((sum, item) => sum + item.close, 0) / window.length;
    return average > 0 ? priceRows[index].close / average - 1 : null;
  });

  return priceRows.map((price, index) => {
    const indicator = indicatorByDate.get(price.date);
    return {
      date: price.date,
      ma60Ratio: Number.isFinite(indicator?.ma_60_ratio ?? NaN) ? indicator?.ma_60_ratio ?? null : ma60Values[index],
      ma20Ratio: Number.isFinite(indicator?.ma_20_ratio ?? NaN) ? indicator?.ma_20_ratio ?? null : ma20Values[index],
      macdRatio: Number.isFinite(indicator?.macd_ratio ?? NaN) ? indicator?.macd_ratio ?? null : null,
      rsi: normalizeRsi(indicator?.rsi),
      atrRatio: Number.isFinite(indicator?.atr_ratio ?? NaN) ? indicator?.atr_ratio ?? null : null,
      bbPosition: Number.isFinite(indicator?.bb_position ?? NaN) ? indicator?.bb_position ?? null : null,
    };
  });
}

function getIndicatorRawTarget(row: ReturnType<typeof buildIndicatorRows>[number]) {
  const ma60 = row.ma60Ratio ?? -Infinity;
  const ma20 = row.ma20Ratio ?? -Infinity;
  const macd = row.macdRatio ?? -Infinity;
  const rsi = row.rsi ?? Infinity;
  const atr = row.atrRatio ?? 0;
  const bb = row.bbPosition ?? Infinity;
  const trendEntry =
    ma60 >= INDICATOR_BASELINE_RULE.ma60Entry &&
    ma20 >= INDICATOR_BASELINE_RULE.ma20Entry &&
    macd >= INDICATOR_BASELINE_RULE.macdEntry &&
    rsi < INDICATOR_BASELINE_RULE.rsiEntryCap;
  const pullbackEntry =
    ma60 >= INDICATOR_BASELINE_RULE.ma60Entry &&
    bb <= INDICATOR_BASELINE_RULE.pullbackBb &&
    rsi < INDICATOR_BASELINE_RULE.pullbackRsi;
  const trendExit = ma60 <= INDICATOR_BASELINE_RULE.ma60Exit || ma20 <= INDICATOR_BASELINE_RULE.ma20Exit;
  const volatilityExit = atr >= INDICATOR_BASELINE_RULE.atrExit && ma20 < 0;

  if (trendEntry) {
    return { target: 1 as const, reason: "60일 추세와 20일 추세가 살아 있고 MACD가 양수라 매수합니다." };
  }
  if (pullbackEntry) {
    return { target: 1 as const, reason: "큰 추세가 살아 있는 상태에서 과열이 낮아 반등 후보로 매수합니다." };
  }
  if (trendExit && volatilityExit) {
    return { target: 0 as const, reason: "추세 약화와 변동성 확대가 함께 나타나 매도합니다." };
  }
  if (trendExit) {
    return { target: 0 as const, reason: "60일 또는 20일 추세가 기준 아래로 약해져 매도합니다." };
  }
  if (volatilityExit) {
    return { target: 0 as const, reason: "변동성이 커지고 단기 추세가 약해져 매도합니다." };
  }
  return { target: 0 as const, reason: "진입 기준이 아직 충분하지 않아 대기합니다." };
}

function buildIndicatorSignals(params: { priceRows: PriceBar[]; indicators: IndicatorPoint[] }) {
  const rows = buildIndicatorRows(params.priceRows, params.indicators).filter(
    (row) => Number.isFinite(row.ma60Ratio ?? NaN) && Number.isFinite(row.ma20Ratio ?? NaN)
  );
  let currentPosition: 0 | 1 = 0;
  let exitStreak = 0;
  let entryStreak = 0;

  return rows.map((row) => {
    const raw = getIndicatorRawTarget(row);
    const lowering = raw.target < currentPosition;
    const raising = raw.target > currentPosition;
    let reason = raw.reason;

    if (lowering) {
      exitStreak += 1;
    } else {
      exitStreak = 0;
    }

    if (raising) {
      entryStreak += 1;
    } else {
      entryStreak = 0;
    }

    if (lowering && exitStreak >= INDICATOR_BASELINE_RULE.exitConfirmDays) {
      currentPosition = raw.target;
      entryStreak = 0;
      reason = `${raw.reason} 청산 조건을 ${INDICATOR_BASELINE_RULE.exitConfirmDays}일 확인했습니다.`;
    } else if (lowering) {
      reason = `${raw.reason} 청산 조건 확인 중이라 직전 포지션을 유지합니다.`;
    } else if (raising && entryStreak >= INDICATOR_BASELINE_RULE.entryConfirmDays) {
      currentPosition = raw.target;
      exitStreak = 0;
      reason = `${raw.reason} 진입 조건을 ${INDICATOR_BASELINE_RULE.entryConfirmDays}일 확인했습니다.`;
    } else if (raising) {
      reason = `${raw.reason} 진입 조건 확인 중이라 현금 대기를 유지합니다.`;
    } else if (!lowering && !raising) {
      currentPosition = raw.target;
    }

    return {
      date: row.date,
      position: currentPosition,
      targetPosition: raw.target,
      conservativeReturn: null,
      lowerBandReturn: null,
      bandWidthReturn: null,
      bandWidthExpansion: null,
      bandWidthPercentile: null,
      ma60Ratio: row.ma60Ratio,
      ma20Ratio: row.ma20Ratio,
      macdRatio: row.macdRatio,
      rsi: row.rsi,
      atrRatio: row.atrRatio,
      reason,
    };
  });
}

function buildSignals(params: {
  priceRows: PriceBar[];
  lineHistory: PredictionResult[];
  bandHistory: PredictionResult[];
}) {
  const rows = buildRawSignals(params);
  let currentPosition: 0 | 1 = 0;
  let exitStreak = 0;
  let entryStreak = 0;

  return rows.map((row) => {
    const raw = getRawTarget(row);
    const lowering = raw.target < currentPosition;
    const raising = raw.target > currentPosition;
    let reason = raw.reason;

    if (lowering) {
      exitStreak += 1;
    } else {
      exitStreak = 0;
    }

    if (raising) {
      entryStreak += 1;
    } else {
      entryStreak = 0;
    }

    if (lowering && exitStreak >= LENS_BALANCE_RULE.confirmDays) {
      currentPosition = raw.target;
      entryStreak = 0;
      reason = `${raw.reason} 청산 조건이 ${LENS_BALANCE_RULE.confirmDays}일 확인되어 매도합니다.`;
    } else if (lowering) {
      reason = `${raw.reason} 청산 조건 확인 중이라 직전 포지션을 유지합니다.`;
    } else if (raising && entryStreak >= LENS_BALANCE_RULE.reentryConfirmDays) {
      currentPosition = raw.target;
      exitStreak = 0;
      reason = `${raw.reason} 재진입 조건이 ${LENS_BALANCE_RULE.reentryConfirmDays}일 확인되어 매수합니다.`;
    } else if (raising) {
      reason = `${raw.reason} 재진입 조건 확인 중이라 현금 대기를 유지합니다.`;
    } else if (!lowering && !raising) {
      currentPosition = raw.target;
    }

    return {
      ...row,
      position: currentPosition,
      targetPosition: raw.target,
      reason,
    };
  });
}

function getLatestIndicatorBefore(indicators: IndicatorPoint[], date: string | null) {
  if (!date) {
    return indicators.at(-1) ?? null;
  }
  for (let index = indicators.length - 1; index >= 0; index -= 1) {
    if (indicators[index].date <= date) {
      return indicators[index];
    }
  }
  return indicators.at(-1) ?? null;
}

function getBandWidthState(card: Pick<StrategySignalCard, "bandWidthExpansion" | "bandWidthPercentile" | "bandWidthReturn">) {
  if (card.bandWidthReturn == null) {
    return "-";
  }
  if (card.bandWidthExpansion != null && card.bandWidthExpansion >= LENS_BALANCE_RULE.widthExpansionThreshold) {
    return "확장";
  }
  if (card.bandWidthPercentile != null && card.bandWidthPercentile >= LENS_BALANCE_RULE.widthPercentileThreshold) {
    return "넓음";
  }
  return "보통";
}

function classifySignalGroup(signal: RiskSignal) {
  const lowerBandReturn = signal.lowerBandReturn ?? Infinity;
  const conservativeReturn = signal.conservativeReturn ?? Infinity;
  const lowerRisky = lowerBandReturn <= LENS_BALANCE_RULE.lowerRiskThreshold;
  const widthExpanded =
    signal.bandWidthExpansion != null && signal.bandWidthExpansion >= LENS_BALANCE_RULE.widthExpansionThreshold;
  const lineWeak = conservativeReturn < LENS_BALANCE_RULE.lineHoldThreshold;

  if (signal.targetPosition === 1 && signal.position === 0) {
    return { group: "buy" as const, label: "매수 후보" };
  }
  if (signal.position === 1) {
    return { group: "hold" as const, label: "전략상 보유 유지" };
  }
  if (lineWeak || lowerRisky || widthExpanded) {
    return { group: "risk" as const, label: "위험 확대" };
  }
  return { group: "watch" as const, label: "관망" };
}

function classifyIndicatorSignal(signal: RiskSignal) {
  const ma60Weak = (signal.ma60Ratio ?? 0) <= INDICATOR_BASELINE_RULE.ma60Exit;
  const ma20Weak = (signal.ma20Ratio ?? 0) <= INDICATOR_BASELINE_RULE.ma20Exit;

  if (signal.targetPosition === 1 && signal.position === 0) {
    return { group: "buy" as const, label: "매수 후보" };
  }
  if (signal.position === 1) {
    return { group: "hold" as const, label: "보유 유지" };
  }
  if (ma60Weak || ma20Weak || signal.reason.includes("변동성")) {
    return { group: "risk" as const, label: "위험 확대" };
  }
  return { group: "watch" as const, label: "관망" };
}

async function loadStrategySignalCard(ticker: string, strategyId: StrategyId): Promise<StrategySignalCard> {
  const baseCard: StrategySignalCard = {
    ticker,
    sector: null,
    group: "watch",
    signalLabel: "아직 신호 없음",
    reason:
      strategyId === "lens_balance_v1"
        ? "가격, 예측선, AI 밴드 history 중 일부가 부족해 최신 전략 신호를 만들 수 없습니다."
        : "가격 또는 보조지표가 부족해 최신 전략 신호를 만들 수 없습니다.",
    asofDate: null,
    conservativeReturn: null,
    lowerBandReturn: null,
    bandWidthReturn: null,
    bandWidthExpansion: null,
    bandWidthPercentile: null,
    ma60Ratio: null,
    ma20Ratio: null,
    macdRatio: null,
    rsi: null,
    atrRatio: null,
    hasUsableSignal: false,
  };

  try {
    const [prices, indicatorsResponse, lineResponse, bandResponse, tickerResponse] = await Promise.all([
      fetchPriceHistory(ticker, "1D"),
      fetchIndicators(ticker, { timeframe: "1D", limit: 300 }).catch(() => null),
      // legacy /predictions/history endpoint 제거됨. lens_balance_v1 은 v1 endpoint 로 옮겨야 한다.
      Promise.resolve(null as { data: PredictionResult[] } | null),
      Promise.resolve(null as { data: PredictionResult[] } | null),
      fetchTickers({ search: ticker, limit: 1 }).catch(() => null),
    ]);
    const indicators = sortUniqueByDate(indicatorsResponse?.data.data ?? []);
    const lineRows = lineResponse?.data ?? [];
    const bandRows = bandResponse?.data ?? [];
    const sector = tickerResponse?.data.find((item) => item.ticker === ticker)?.sector ?? null;

    if (prices.length === 0) {
      return { ...baseCard, sector };
    }

    const signals =
      strategyId === "lens_balance_v1"
        ? lineRows.length > 0 && bandRows.length > 0
          ? buildSignals({ priceRows: prices, lineHistory: lineRows, bandHistory: bandRows })
          : []
        : buildIndicatorSignals({ priceRows: prices, indicators });
    const latestSignal = signals.at(-1);
    if (!latestSignal) {
      return { ...baseCard, sector };
    }

    const latestIndicator = getLatestIndicatorBefore(indicators, latestSignal.date);
    const classification = strategyId === "lens_balance_v1" ? classifySignalGroup(latestSignal) : classifyIndicatorSignal(latestSignal);
    return {
      ticker,
      sector,
      group: classification.group,
      signalLabel: classification.label,
      reason: latestSignal.reason,
      asofDate: latestSignal.date,
      conservativeReturn: latestSignal.conservativeReturn,
      lowerBandReturn: latestSignal.lowerBandReturn,
      bandWidthReturn: latestSignal.bandWidthReturn,
      bandWidthExpansion: latestSignal.bandWidthExpansion,
      bandWidthPercentile: latestSignal.bandWidthPercentile,
      ma60Ratio: latestSignal.ma60Ratio ?? latestIndicator?.ma_60_ratio ?? null,
      ma20Ratio: latestSignal.ma20Ratio ?? latestIndicator?.ma_20_ratio ?? null,
      macdRatio: latestSignal.macdRatio ?? latestIndicator?.macd_ratio ?? null,
      rsi: latestSignal.rsi ?? normalizeRsi(latestIndicator?.rsi),
      atrRatio: latestSignal.atrRatio ?? latestIndicator?.atr_ratio ?? null,
      hasUsableSignal: true,
    };
  } catch {
    return baseCard;
  }
}

function calculateMaxDrawdown(points: BacktestPoint[], equityKey: "strategyEquity" | "buyHoldEquity" = "strategyEquity") {
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

function calculateSharpe(returns: number[]) {
  if (returns.length < 2) {
    return 0;
  }
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (returns.length - 1);
  const std = Math.sqrt(variance);
  return std > 0 ? (mean / std) * Math.sqrt(252) : 0;
}

function calculateSortino(returns: number[]) {
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

function quantile(values: number[], ratio: number) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * ratio)));
  return sorted[index];
}

function chooseLargeLossThreshold(returns: number[]) {
  const lowerQuintile = quantile(returns, 0.2);
  if (lowerQuintile == null) {
    return null;
  }
  return Math.min(-0.02, lowerQuintile);
}

function runStrategyBacktest(params: {
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

function buildPath(values: Array<{ x: number; y: number }>) {
  return values.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
}

function MiniLineChart({
  title,
  lines,
  markers,
}: {
  title: string;
  lines: LineSeries[];
  markers?: TradeEvent[];
}) {
  const width = 720;
  const height = 220;
  const padding = { top: 18, right: 18, bottom: 28, left: 42 };
  const allValues = lines.flatMap((line) => line.values.map((point) => point.value).filter(Number.isFinite));
  const longest = lines.reduce((current, line) => (line.values.length > current.length ? line.values : current), [] as LineSeries["values"]);

  if (allValues.length < 2 || longest.length < 2) {
    return <div className="empty-state">{title}를 표시할 데이터가 없습니다.</div>;
  }

  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const span = Math.max(maxValue - minValue, Math.abs(maxValue) * 0.02, 1);
  const xForIndex = (index: number, length: number) =>
    padding.left + (index / Math.max(length - 1, 1)) * (width - padding.left - padding.right);
  const yForValue = (value: number) => padding.top + ((maxValue - value) / span) * (height - padding.top - padding.bottom);

  return (
    <div className="risk-chart">
      <div className="risk-chart__header">
        <strong>{title}</strong>
        <span>
          {longest[0]?.date} - {longest[longest.length - 1]?.date}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
        {lines.map((line) => {
          const points = line.values.map((point, index) => ({
            x: xForIndex(index, line.values.length),
            y: yForValue(point.value),
          }));
          return <path key={line.label} d={buildPath(points)} stroke={line.color} />;
        })}
        {markers?.map((marker) => {
          const index = longest.findIndex((point) => point.date >= marker.date);
          if (index < 0) {
            return null;
          }
          const x = xForIndex(index, longest.length);
          const priceLine = lines[0];
          const markerPoint = priceLine?.values[index];
          if (!markerPoint) {
            return null;
          }
          return (
            <circle
              key={`${marker.kind}-${marker.date}-${marker.price}`}
              cx={x}
              cy={yForValue(markerPoint.value)}
              r="4"
              className={marker.kind === "entry" ? "risk-chart__marker-entry" : "risk-chart__marker-exit"}
            />
          );
        })}
      </svg>
      <div className="risk-chart__legend">
        {lines.map((line) => (
          <span key={line.label}>
            <i style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
        {markers && markers.length > 0 ? <span>마커: 매수/매도</span> : null}
      </div>
    </div>
  );
}

function PositionStrip({ points }: { points: BacktestPoint[] }) {
  const sampled = points.length > 120 ? points.filter((_, index) => index % Math.ceil(points.length / 120) === 0) : points;
  return (
    <div className="position-strip" aria-label="보유와 현금 구간">
      {sampled.map((point) => (
        <span key={point.date} className={point.position === 1 ? "is-hold" : "is-cash"} title={`${point.date} ${point.position === 1 ? "보유" : "현금"}`} />
      ))}
    </div>
  );
}

export default function BacktestView() {
  const [tickerInput, setTickerInput] = useState("AAPL");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const timeframe: DisplayTimeframe = "1D";
  const [strategyId, setStrategyId] = useState<StrategyId>("indicator_balance_v2");
  const [priceData, setPriceData] = useState<PriceBar[]>([]);
  const [indicatorData, setIndicatorData] = useState<IndicatorPoint[]>([]);
  const [lineHistory, setLineHistory] = useState<PredictionResult[]>([]);
  const [bandHistory, setBandHistory] = useState<PredictionResult[]>([]);
  const [result, setResult] = useState<BacktestSimulationResult | null>(null);
  const [suggestions, setSuggestions] = useState<StockSummary[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchErrorMessage, setSearchErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [signalCards, setSignalCards] = useState<StrategySignalCard[]>([]);
  const [signalLoading, setSignalLoading] = useState(false);
  const [signalErrorMessage, setSignalErrorMessage] = useState<string | null>(null);
  const [expandedSignalGroups, setExpandedSignalGroups] = useState<Partial<Record<SignalGroupId, boolean>>>({});
  const detailRef = useRef<HTMLDivElement | null>(null);
  const deferredTicker = useDeferredValue(tickerInput);

  useEffect(() => {
    const keyword = deferredTicker.trim();
    if (keyword.length < 1) {
      setSuggestions([]);
      setSearchErrorMessage(null);
      return;
    }

    let active = true;
    setSearchLoading(true);
    setSearchErrorMessage(null);
    fetchTickers({ search: keyword, limit: 6 })
      .then((response) => {
        if (active) {
          setSuggestions(response.data);
        }
      })
      .catch(() => {
        if (active) {
          setSuggestions([]);
          setSearchErrorMessage("티커 검색을 사용할 수 없습니다. 티커를 직접 입력하면 백테스트 조회는 계속 가능합니다.");
        }
      })
      .finally(() => {
        if (active) {
          setSearchLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [deferredTicker]);

  useEffect(() => {
    let active = true;
    setSignalLoading(true);
    setSignalErrorMessage(null);
    const signalPromise = isBackendStrategy(strategyId)
      ? fetchStrategyScan(strategyId, { limit: 500 }).then((response) => response.data.cards)
      : Promise.all(SIGNAL_SCAN_TICKERS.map((ticker) => loadStrategySignalCard(ticker, strategyId)));

    signalPromise
      .then((cards) => {
        if (active) {
          setSignalCards(cards);
        }
      })
      .catch(() => {
        if (active) {
          setSignalCards([]);
          setSignalErrorMessage("전략 신호를 불러오지 못했습니다. 직접 종목 확인은 계속 사용할 수 있습니다.");
        }
      })
      .finally(() => {
        if (active) {
          setSignalLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [strategyId]);

  async function loadBacktest(nextTicker: string, nextTimeframe: DisplayTimeframe, nextStrategyId = strategyId) {
    setIsLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    setResult(null);
    setLineHistory([]);
    setBandHistory([]);

    try {
      const normalizedTicker = nextTicker.trim().toUpperCase() || "AAPL";
      if (isBackendStrategy(nextStrategyId)) {
        const response = await fetchStrategyBacktest(nextStrategyId, normalizedTicker);
        const nextResult = response.data;
        setResult(nextResult);
        setPriceData(
          nextResult.points.map((point) => ({
            date: point.date,
            open: point.price,
            high: point.price,
            low: point.price,
            close: point.price,
            volume: null,
          }))
        );
        setIndicatorData([]);
        setLineHistory([]);
        setBandHistory([]);
        setStatusMessage(null);
        return;
      }
      const [prices, indicatorsResponse] = await Promise.all([
        fetchPriceHistory(normalizedTicker, nextTimeframe),
        fetchIndicators(normalizedTicker, { timeframe: nextTimeframe, limit: 300 }).catch(() => null),
      ]);
      const indicators = sortUniqueByDate(indicatorsResponse?.data.data ?? []);
      setPriceData(prices);
      setIndicatorData(indicators);

      if (prices.length === 0) {
        setStatusMessage("가격 데이터가 없습니다. 가격 데이터가 연결되면 전략 결과와 차트를 표시합니다.");
        return;
      }

      if (nextTimeframe !== "1D") {
        setStatusMessage(`${getStrategyDefinition(nextStrategyId).label}은 1D 제품 line/band layer로만 계산합니다. ${nextTimeframe}에서는 가격 데이터만 확인할 수 있습니다.`);
        return;
      }

      const needsLine = strategyNeedsLine(nextStrategyId);
      const needsBand = strategyNeedsBand(nextStrategyId);
      // legacy /predictions/history endpoint 제거됨. lens_balance_v1 류 strategy 는 v1 endpoint 로 옮겨야 한다.
      const [lineResponse, bandResponse] = await Promise.all([
        Promise.resolve(null as { data: PredictionResult[] } | null),
        Promise.resolve(null as { data: PredictionResult[] } | null),
      ]);
      const lineRows = lineResponse?.data ?? [];
      const bandRows = bandResponse?.data ?? [];
      setLineHistory(lineRows);
      setBandHistory(bandRows);

      if ((needsLine && lineRows.length === 0) || (needsBand && bandRows.length === 0)) {
        const missingParts = [
          needsLine && lineRows.length === 0 ? "보수적 기준선" : null,
          needsBand && bandRows.length === 0 ? "AI 밴드" : null,
        ].filter(Boolean);
        setStatusMessage(`이 티커에는 아직 ${missingParts.join(", ")} 저장 결과가 없습니다. 가격 데이터만 표시합니다.`);
        return;
      }

      const nextResult = runStrategyBacktest({
        strategyId: nextStrategyId,
        priceRows: prices,
        lineHistory: lineRows,
        bandHistory: bandRows,
        indicators,
        feeBps: DEFAULT_FEE_BPS,
      });
      setResult(nextResult);
      setStatusMessage(nextResult ? null : "전략 계산에 필요한 날짜가 맞지 않아 결과를 계산할 수 없습니다.");
    } catch (error) {
      setPriceData([]);
      setIndicatorData([]);
      setResult(null);
      setErrorMessage(extractErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadBacktest("AAPL", "1D", "indicator_balance_v2");
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextTicker = tickerInput.trim().toUpperCase() || "AAPL";
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadBacktest(nextTicker, timeframe, strategyId);
    });
    window.setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
  }

  function handleTickerSelect(nextTicker: string, scrollToDetail = true) {
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadBacktest(nextTicker, timeframe, strategyId);
    });
    if (scrollToDetail) {
      window.setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
    }
  }

  function handleStrategyChange(nextStrategyId: StrategyId) {
    setStrategyId(nextStrategyId);
    startTransition(() => {
      void loadBacktest(selectedTicker, timeframe, nextStrategyId);
    });
  }

  function toggleSignalGroup(groupId: SignalGroupId) {
    setExpandedSignalGroups((current) => ({
      ...current,
      [groupId]: !current[groupId],
    }));
  }

  const priceSeries = useMemo(
    () =>
      priceData.map((row) => ({
        date: row.date,
        value: row.close,
      })),
    [priceData]
  );
  const strategyEquitySeries = useMemo(
    () => result?.points.map((point) => ({ date: point.date, value: (point.strategyEquity - 1) * 100 })) ?? [],
    [result]
  );
  const buyHoldEquitySeries = useMemo(
    () => result?.points.map((point) => ({ date: point.date, value: (point.buyHoldEquity - 1) * 100 })) ?? [],
    [result]
  );
  const priceByDate = useMemo(() => new Map(priceData.map((row) => [row.date, row])), [priceData]);
  const tradeRecords = useMemo<TradeRecord[]>(() => {
    if (!result) {
      return [];
    }
    let previousPosition: 0 | 1 = 0;
    return result.signals.map((signal) => {
      const priceIndex = priceData.findIndex((row) => row.date === signal.date);
      const price = priceIndex >= 0 ? priceData[priceIndex] : priceByDate.get(signal.date);
      const nextPrice = priceIndex >= 0 ? priceData[priceIndex + 1] : undefined;
      const action =
        signal.position === 1 && previousPosition === 0
          ? "매수"
          : signal.position === 0 && previousPosition === 1
            ? "매도"
            : signal.position === 1
              ? "보유"
              : "대기";
      previousPosition = signal.position;
      return {
        date: signal.date,
        action,
        price: price?.close ?? null,
        reason: signal.reason,
        nextDayReturn: price && nextPrice ? nextPrice.close / price.close - 1 : null,
        conservativeReturn: signal.conservativeReturn,
        lowerBandReturn: signal.lowerBandReturn,
        bandWidthReturn: signal.bandWidthReturn,
        bandWidthExpansion: signal.bandWidthExpansion,
        bandWidthPercentile: signal.bandWidthPercentile,
        ma60Ratio: signal.ma60Ratio,
        ma20Ratio: signal.ma20Ratio,
        macdRatio: signal.macdRatio,
        rsi: signal.rsi,
        atrRatio: signal.atrRatio,
      };
    });
  }, [priceData, priceByDate, result]);
  const executionRecords = tradeRecords.filter((record) => record.action === "매수" || record.action === "매도");
  const latestTradeRecord = tradeRecords.length > 0 ? tradeRecords[tradeRecords.length - 1] : null;
  const strategyDefinition = getStrategyDefinition(strategyId);
  const groupedSignalCards = useMemo(
    () =>
      SIGNAL_GROUPS.map((group) => {
        const allCards = signalCards.filter((card) => card.group === group.id);
        const isExpanded = Boolean(expandedSignalGroups[group.id]);
        const visibleLimit = isExpanded ? SIGNAL_GROUP_MAX_LIMIT : SIGNAL_GROUP_DEFAULT_LIMIT[group.id];
        return {
          ...group,
          allCards,
          cards: allCards.slice(0, visibleLimit),
          isExpanded,
          canToggle: allCards.length > SIGNAL_GROUP_DEFAULT_LIMIT[group.id] || group.id === "watch",
        };
      }),
    [expandedSignalGroups, signalCards]
  );
  const usableSignalCount = signalCards.filter((card) => card.hasUsableSignal).length;
  const scanScopeCount = isBackendStrategy(strategyId) ? signalCards.length : SIGNAL_SCAN_TICKERS.length;
  const selectedSignalCard = signalCards.find((card) => card.ticker === selectedTicker);
  const latestSignalDate = signalCards
    .map((card) => card.asofDate)
    .filter((date): date is string => Boolean(date))
    .sort()
    .at(-1);
  const visibleDecisionFactors = latestTradeRecord
    ? [
        {
          id: "conservative",
          label: "보수적 기준선",
          value: formatRatioAsPercent(latestTradeRecord.conservativeReturn),
          description: "-0.2% 이상이면 진입 후보, -1.4% 이상이면 보유 후보로 봅니다.",
        },
        {
          id: "lowerBand",
          label: "AI 밴드 하단 위험",
          value: formatRatioAsPercent(latestTradeRecord.lowerBandReturn),
          description: "-5% 이하이면 하방 위험이 깊어진 상태로 봅니다.",
        },
        {
          id: "bandWidth",
          label: "AI 밴드 폭",
          value: formatUnsignedRatioAsPercent(latestTradeRecord.bandWidthReturn),
          description: "AI가 보는 예상 변동 범위의 넓이입니다.",
        },
        {
          id: "bandExpansion",
          label: "밴드 폭 확장",
          value: latestTradeRecord.bandWidthExpansion == null ? "-" : `${formatNumber(latestTradeRecord.bandWidthExpansion, 2)}배`,
          description: "최근 20개 신호의 밴드 폭 중앙값 대비 1.10배 이상이면 신규 진입을 막습니다.",
        },
        {
          id: "bandPercentile",
          label: "밴드 폭 위치",
          value: latestTradeRecord.bandWidthPercentile == null ? "-" : formatUnsignedRatioAsPercent(latestTradeRecord.bandWidthPercentile),
          description: "최근 history 안에서 밴드 폭이 어느 정도 넓은 편인지 보는 참고값입니다.",
        },
        {
          id: "ma60Trend",
          label: "60일 추세",
          value: formatRatioAsPercent(latestTradeRecord.ma60Ratio),
          description: "+2% 이상이면 상승 추세로 보고, -5% 이하이면 방어 후보로 봅니다.",
        },
        {
          id: "ma20Trend",
          label: "20일 추세",
          value: formatRatioAsPercent(latestTradeRecord.ma20Ratio),
          description: "단기 흐름이 크게 무너지면 청산 후보로 봅니다.",
        },
        {
          id: "macd",
          label: "MACD",
          value: formatRatioAsPercent(latestTradeRecord.macdRatio),
          description: "0보다 크면 추세 모멘텀이 살아 있는 쪽으로 봅니다.",
        },
        {
          id: "rsi",
          label: "RSI",
          value: formatNumber(latestTradeRecord.rsi, 1),
          description: "75 이상이면 신규 진입을 조심합니다.",
        },
        {
          id: "atr",
          label: "ATR",
          value: formatRatioAsPercent(latestTradeRecord.atrRatio),
          description: "변동성이 커질 때 포지션 위험을 확인합니다.",
        },
      ].filter((factor) => strategyDefinition.visibleFactors.includes(factor.id as DecisionFactorId))
    : [];

  return (
    <div className="view-stack">
      <header className="view-header">
        <div className="view-header__title">
          <div className="eyebrow">백테스트</div>
          <h1>전략 신호</h1>
          <p>전략과 티커를 먼저 선택하고, 아래쪽 500티커 신호판은 같은 룰을 각 티커에 독립 적용한 최신 상태로 확인합니다.</p>
        </div>
        <div className="status-badge status-badge--neutral">{strategyDefinition.label}</div>
      </header>

      <section className="panel strategy-signal-overview">
        <div>
          <div className="eyebrow">오늘의 신호 요약</div>
          <h2>{strategyDefinition.label} 기준 신호</h2>
          <p>
            {strategyDefinition.usesAi
              ? "저장된 1D 제품 예측선과 AI 밴드 history가 있는 종목만 읽어 최신 전략 상태를 요약합니다."
              : "AI 예측 없이 가격과 보조지표만 읽어 최신 전략 상태를 요약합니다."}
            {" "}포트폴리오 추천이 아니라 단일 티커 백테스트로 이어지는 신호 탐색 화면입니다.
          </p>
          <p className="strategy-signal-scope-note">
            {isBackendStrategy(strategyId)
              ? "현재 백엔드 로컬 parquet 기준 단일 티커 long/cash 신호입니다. 500개 내외 티커에 같은 룰을 각각 적용하고, 종목을 선택하면 그 티커 하나의 상세 백테스트를 봅니다."
              : "현재는 12개 주요 종목 기준 전략 신호입니다. 전체 scanner API가 붙기 전까지는 subset 신호로만 봅니다."}
          </p>
        </div>
        <div className="strategy-signal-summary">
          <div>
            <span>스캔 범위</span>
            <strong>{scanScopeCount}개</strong>
          </div>
          <div>
            <span>신호 확인</span>
            <strong>{signalLoading ? "조회 중" : `${usableSignalCount}개`}</strong>
          </div>
          <div>
            <span>기준일</span>
            <strong>{latestSignalDate ?? "-"}</strong>
          </div>
          <div>
            <span>전략 단위</span>
            <strong>1D</strong>
          </div>
        </div>
      </section>

      {signalErrorMessage ? <div className="notice notice--error">{signalErrorMessage}</div> : null}

      <section className="strategy-signal-board" aria-label="Lens Balance v1 전략 신호 그룹">
        {groupedSignalCards.map((group) => (
          <section className={`panel signal-group-panel signal-group-panel--${group.id}`} key={group.id}>
            <div className="panel-heading">
              <div className="eyebrow">전략 신호</div>
              <div className="signal-group-heading-row">
                <h2>{group.title}</h2>
                <span>{group.allCards.length}개</span>
              </div>
              <p>{group.description}</p>
            </div>
            <div className="signal-card-list">
              {signalLoading ? (
                <div className="compact-note">신호를 불러오는 중입니다.</div>
              ) : group.cards.length > 0 ? (
                group.cards.map((card) => (
                  <button
                    type="button"
                    key={`${group.id}-${card.ticker}`}
                    className={`signal-card signal-card--${group.id} ${selectedTicker === card.ticker ? "is-selected" : ""}`}
                    onClick={() => handleTickerSelect(card.ticker)}
                    aria-pressed={selectedTicker === card.ticker}
                  >
                    <div className="signal-card__top">
                      <strong>{card.ticker}</strong>
                      <span>{card.sector ?? "종목명 없음"}</span>
                    </div>
                    <div className="signal-card__label">{card.signalLabel}</div>
                    <dl>
                      {strategyNeedsLine(strategyId) ? (
                        <>
                          <div>
                            <dt>예측선</dt>
                            <dd>{formatRatioAsPercent(card.conservativeReturn)}</dd>
                          </div>
                          <div>
                            <dt>하단 위험</dt>
                            <dd>{formatRatioAsPercent(card.lowerBandReturn)}</dd>
                          </div>
                          <div>
                            <dt>밴드 폭</dt>
                            <dd>{getBandWidthState(card)}</dd>
                          </div>
                          <div>
                            <dt>60일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma60Ratio)}</dd>
                          </div>
                        </>
                      ) : strategyNeedsBand(strategyId) ? (
                        <>
                          <div>
                            <dt>하단 위험</dt>
                            <dd>{formatRatioAsPercent(card.lowerBandReturn)}</dd>
                          </div>
                          <div>
                            <dt>밴드 폭</dt>
                            <dd>{getBandWidthState(card)}</dd>
                          </div>
                          <div>
                            <dt>60일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma60Ratio)}</dd>
                          </div>
                          <div>
                            <dt>RSI</dt>
                            <dd>{formatNumber(card.rsi, 1)}</dd>
                          </div>
                        </>
                      ) : (
                        <>
                          <div>
                            <dt>60일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma60Ratio)}</dd>
                          </div>
                          <div>
                            <dt>20일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma20Ratio)}</dd>
                          </div>
                          <div>
                            <dt>RSI</dt>
                            <dd>{formatNumber(card.rsi, 1)}</dd>
                          </div>
                          <div>
                            <dt>ATR</dt>
                            <dd>{formatRatioAsPercent(card.atrRatio)}</dd>
                          </div>
                        </>
                      )}
                    </dl>
                    <p>{card.hasUsableSignal ? card.reason : "아직 신호 없음"}</p>
                  </button>
                ))
              ) : group.id === "watch" && group.allCards.length > 0 ? (
                <div className="compact-note">관망 종목은 접혀 있습니다.</div>
              ) : (
                <div className="empty-state">아직 신호 없음</div>
              )}
            </div>
            {!signalLoading && group.canToggle ? (
              <button type="button" className="signal-group-toggle" onClick={() => toggleSignalGroup(group.id)}>
                {group.isExpanded ? "접기" : group.id === "watch" ? "관망 보기" : `더 보기 (${Math.min(group.allCards.length, SIGNAL_GROUP_MAX_LIMIT)}개)`}
              </button>
            ) : null}
          </section>
        ))}
      </section>

      <section className="panel direct-ticker-panel">
        <div className="panel-heading">
          <div className="eyebrow">전략 및 티커 검색</div>
          <h2>먼저 종목을 직접 확인</h2>
          <p>AAPL처럼 현재 상위 후보에 없는 종목도 직접 조회해 전략 편입 여부와 상세 백테스트를 확인합니다.</p>
        </div>
        <form className="backtest-toolbar backtest-toolbar--compact" onSubmit={handleSubmit}>
          <label className="search-field">
            <span>티커</span>
            <input value={tickerInput} onChange={(event) => setTickerInput(event.target.value.toUpperCase())} />
          </label>
          <label className="select-field">
            <span>전략</span>
            <select value={strategyId} onChange={(event) => handleStrategyChange(event.target.value as StrategyId)}>
              {STRATEGIES.map((strategy) => (
                <option key={strategy.id} value={strategy.id}>
                  {strategy.label}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" className="primary-button primary-button--compact" disabled={isLoading}>
            조회
          </button>
          <div className="fee-pill">
            <span>전략</span>
            <strong>{strategyDefinition.label}</strong>
            <em>1D / {formatNumber(DEFAULT_FEE_BPS, 1)}bp</em>
          </div>
        </form>

        {suggestions.length > 0 ? (
          <div className="suggestion-row">
            {suggestions.map((item) => (
              <button key={item.ticker} type="button" onClick={() => handleTickerSelect(item.ticker)}>
                <strong>{item.ticker}</strong>
                <span>{item.sector ?? "섹터 없음"}</span>
              </button>
            ))}
          </div>
        ) : searchLoading ? (
          <div className="compact-note">검색 중입니다.</div>
        ) : null}
        {searchErrorMessage ? <div className="compact-note compact-note--warning">{searchErrorMessage}</div> : null}
      </section>

      <section className="panel strategy-rule-panel">
        <div>
          <div className="eyebrow">전략 설명</div>
          <h2>{strategyDefinition.label}</h2>
          <p>{strategyDefinition.description}</p>
          <p className="compact-note">
            이 화면은 매수·매도 추천이 아니라 룰 기반 시뮬레이션입니다. 현금 보유는 전략 기준이 약할 때 시장 참여를 잠시 줄이는 방식입니다.
          </p>
          <p className="compact-note compact-note--warning">{strategyDefinition.scopeNote}</p>
          <div className="strategy-validation-strip" aria-label={`${strategyDefinition.label} 검증 결과`}>
            {strategyDefinition.validationRows.map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </div>
        <div className="rule-grid">
          {strategyDefinition.ruleRows.map(([label, value]) => (
            <Fragment key={`${label}-${value}`}>
              <span>{label}</span>
              <strong>{value}</strong>
            </Fragment>
          ))}
        </div>
      </section>

      {errorMessage ? <div className="notice notice--error">{errorMessage}</div> : null}
      {statusMessage ? <div className="notice">{statusMessage}</div> : null}

      <div className="backtest-detail-anchor" ref={detailRef}>
        <section className="panel selected-backtest-heading">
          <div>
            <div className="eyebrow">상세 백테스트</div>
            <h2>{selectedTicker} 단일 티커 상세</h2>
            <p>
              {selectedSignalCard
                ? `${selectedSignalCard.signalLabel} · 기준일 ${selectedSignalCard.asofDate ?? "-"}`
                : "직접 입력한 종목입니다. 저장된 예측이 있으면 전략 결과를 계산합니다."}
            </p>
          </div>
          <div className="status-badge status-badge--neutral">{result ? "상세 계산 완료" : isLoading ? "계산 중" : "결과 대기"}</div>
        </section>

        <section className="metric-grid metric-grid--six">
          <MetricCard label="전략 수익률" value={formatPercent(result?.strategyReturnPct)} />
          <MetricCard label="단순 보유 수익률" value={formatPercent(result?.buyHoldReturnPct)} />
          <MetricCard label="전략 MDD" value={formatPercent(result?.maxDrawdownPct)} tone="bad" />
          <MetricCard label="단순 보유 MDD" value={formatPercent(result?.buyHoldMaxDrawdownPct)} tone="bad" />
          <MetricCard label="손실 회피율" value={result?.largeLossAvoidanceRate == null ? "-" : formatUnsignedRatioAsPercent(result.largeLossAvoidanceRate)} />
          <MetricCard label="시장 참여율" value={result ? formatUnsignedRatioAsPercent(result.marketParticipationRate) : "-"} />
        </section>
      </div>

      <section className="metric-grid metric-grid--six">
        <MetricCard label="전략 Sharpe" value={formatNumber(result?.feeAdjustedSharpe)} />
        <MetricCard label="단순 보유 Sharpe" value={formatNumber(result?.buyHoldSharpe)} />
        <MetricCard label="MDD 개선폭" value={formatPercent(result?.maxDrawdownImprovementPct)} />
        <MetricCard label="거래 횟수" value={formatNumber(result?.tradeCount, 0)} />
        <MetricCard label="평균 보유 기간" value={result?.averageHoldingDays == null ? "-" : `${formatNumber(result.averageHoldingDays, 1)}일`} />
        <MetricCard label="큰 하락 기준" value={formatPercent(result?.largeLossThresholdPct)} />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div className="eyebrow">전략 평가</div>
          <h2>{strategyDefinition.label} 현재 티커 기준 점검</h2>
        </div>
        <div className="strategy-assessment-grid">
          <div>
            <span>수익 추종</span>
            <strong>{evaluateFollowReturn(result)}</strong>
          </div>
          <div>
            <span>낙폭 완화</span>
            <strong>{evaluateDrawdown(result)}</strong>
          </div>
          <div>
            <span>손실 회피</span>
            <strong>{evaluateLossAvoidance(result)}</strong>
          </div>
          <div>
            <span>거래 빈도</span>
            <strong>{evaluateTradeFrequency(result)}</strong>
          </div>
        </div>
      </section>

      <section className="chart-grid">
        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">가격과 매매</div>
            <h2>{selectedTicker} 가격 차트</h2>
          </div>
          <MiniLineChart
            title="가격"
            lines={[{ label: selectedTicker, color: "#1f2937", values: priceSeries }]}
            markers={result?.tradeEvents}
          />
          {result ? (
            <details className="backtest-detail">
              <summary>매매 구간 보기</summary>
              <PositionStrip points={result.points} />
              <div className="position-strip__legend">
                <span><i className="is-hold" />보유 구간</span>
                <span><i className="is-cash" />현금 대기 구간</span>
                <span>가격 차트의 점은 매수/매도 시점입니다.</span>
              </div>
              {result.tradeEvents.length > 0 ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>날짜</th>
                        <th>행동</th>
                        <th>가격</th>
                        <th>이유</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.tradeEvents.slice(-10).reverse().map((event) => (
                        <tr key={`${event.kind}-${event.date}-${event.price}`}>
                          <td>{event.date}</td>
                          <td>{event.kind === "entry" ? "매수" : "매도"}</td>
                          <td>{formatNumber(event.price, 2)}</td>
                          <td>{event.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state">이 기간에는 매수/매도 전환이 없습니다.</div>
              )}
            </details>
          ) : null}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">성과 비교</div>
            <h2>전략 vs 단순 보유</h2>
          </div>
          {result ? (
            <MiniLineChart
              title="누적 수익률"
              lines={[
                { label: strategyDefinition.shortLabel, color: "#0f766e", values: strategyEquitySeries },
                { label: "단순 보유", color: "#64748b", values: buyHoldEquitySeries },
              ]}
            />
          ) : (
            <div className="empty-state">{isLoading ? "조회 중입니다." : "전략 결과를 계산할 수 없습니다."}</div>
          )}
        </div>
      </section>

      <section className="chart-grid">
        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">전략 판단 요소</div>
            <h2>최근 판단 입력값</h2>
          </div>
          {latestTradeRecord ? (
            <div className="decision-factor-grid">
              {visibleDecisionFactors.map((factor) => (
                <div key={factor.id}>
                  <span>{factor.label}</span>
                  <strong>{factor.value}</strong>
                  <p>{factor.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">표시할 전략 판단 요소가 없습니다.</div>
          )}
        </div>
        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">매매 기록</div>
            <h2>최근 매매 기록</h2>
          </div>
          {executionRecords.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>날짜</th>
                    <th>행동</th>
                    <th>가격</th>
                    <th>이유</th>
                    <th>다음 날 수익률</th>
                  </tr>
                </thead>
                <tbody>
                  {executionRecords.slice(-10).reverse().map((record) => (
                    <tr key={`${record.date}-${record.action}`}>
                      <td>{record.date}</td>
                      <td>{record.action}</td>
                      <td>{formatNumber(record.price, 2)}</td>
                      <td>{record.reason}</td>
                      <td>{formatRatioAsPercent(record.nextDayReturn)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">이 기간에는 매수/매도 전환이 없습니다.</div>
          )}
        </div>
      </section>
    </div>
  );
}
