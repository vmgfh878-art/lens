"use client";

import { FormEvent, Fragment, startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import {
  DisplayTimeframe,
  fetchIndicators,
  fetchPredictionHistory,
  fetchPrices,
  fetchTickers,
  IndicatorPoint,
  PredictionResult,
  PriceBar,
  StockSummary,
} from "@/api/client";
import MetricCard from "@/components/MetricCard";

const PRODUCT_LINE_RUN_ID = "patchtst-1D-efad3c29d803";
const PRODUCT_BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8";
const DEFAULT_PRICE_LOOKBACK_DAYS = 365;
const PREDICTION_HISTORY_LIMIT = 200;
const DEFAULT_FEE_BPS = 10;
const SIGNAL_SCAN_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "JPM", "XOM"];

const LENS_BALANCE_RULE = {
  lineEntryThreshold: -0.002,
  lineHoldThreshold: -0.014,
  lowerRiskThreshold: -0.05,
  widthExpansionThreshold: 1.1,
  widthPercentileThreshold: 0.75,
  confirmDays: 2,
  reentryConfirmDays: 2,
};

const LENS_BALANCE_VALIDATION = [
  ["전략 수익률", "3.98%"],
  ["단순 보유 수익률", "2.37%"],
  ["전략 MDD", "-5.93%"],
  ["단순 보유 MDD", "-13.08%"],
  ["손실 회피율", "55.6%"],
  ["시장 참여율", "44.7%"],
  ["강한 상승 방어율", "67.1%"],
  ["검증 티커", "95개"],
] as const;

type StrategyId = "lens_balance_v1";
type SignalGroupId = "buy" | "hold" | "risk" | "watch";

interface StrategyDefinition {
  id: StrategyId;
  label: string;
  shortLabel: string;
  description: string;
  ruleRows: Array<[string, string]>;
  visibleFactors: Array<"conservative" | "lowerBand" | "bandWidth" | "bandExpansion" | "bandPercentile">;
}

interface RiskSignal {
  date: string;
  position: 0 | 1;
  targetPosition: 0 | 1;
  conservativeReturn: number;
  lowerBandReturn: number;
  bandWidthReturn: number;
  bandWidthExpansion: number | null;
  bandWidthPercentile: number | null;
  reason: string;
}

interface BacktestPoint {
  date: string;
  price: number;
  strategyEquity: number;
  buyHoldEquity: number;
  position: 0 | 1;
}

interface TradeEvent {
  date: string;
  kind: "entry" | "exit";
  price: number;
  reason: string;
}

interface BacktestSimulationResult {
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

interface LineSeries {
  label: string;
  color: string;
  values: Array<{ date: string; value: number }>;
}

interface TradeRecord {
  date: string;
  action: "매수" | "보유" | "매도" | "대기";
  price: number | null;
  reason: string;
  nextDayReturn: number | null;
  conservativeReturn: number;
  lowerBandReturn: number;
  bandWidthReturn: number;
  bandWidthExpansion: number | null;
  bandWidthPercentile: number | null;
}

interface StrategySignalCard {
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
  rsi: number | null;
  hasUsableSignal: boolean;
}

const SIGNAL_GROUPS: Array<{ id: SignalGroupId; title: string; description: string }> = [
  { id: "buy", title: "매수 후보", description: "AI 지표 기준으로 신규 진입을 검토할 수 있는 종목입니다." },
  { id: "hold", title: "보유 유지", description: "전략상 보유 상태를 이어가는 종목입니다." },
  { id: "risk", title: "위험 확대", description: "line 약화나 밴드 위험 확대로 방어 확인이 필요한 종목입니다." },
  { id: "watch", title: "관망", description: "진입 기준이 아직 충분하지 않거나 신호가 부족한 종목입니다." },
];

const SIGNAL_GROUP_DEFAULT_LIMIT: Record<SignalGroupId, number> = {
  buy: 3,
  hold: 3,
  risk: 3,
  watch: 0,
};

const SIGNAL_GROUP_MAX_LIMIT = 10;

const STRATEGIES: StrategyDefinition[] = [
  {
    id: "lens_balance_v1",
    label: "Lens Balance v1",
    shortLabel: "Lens Balance",
    description:
      "보수적 예측선으로 방향을 보고, AI 밴드로 불확실성과 하방 위험을 확인해 진입과 청산을 조절하는 단일 티커 실험 전략입니다.",
    ruleRows: [
      ["사용 지표", "보수적 예측선 v1, AI 밴드 v1"],
      ["예측선 버전", `보수적 예측선 v1 (${PRODUCT_LINE_RUN_ID})`],
      ["밴드 버전", `AI 밴드 v1 (${PRODUCT_BAND_RUN_ID})`],
      ["진입", "보수적 예측선 >= -0.2%, 밴드 폭 급확장 아님"],
      ["보유", "보수적 예측선 >= -1.4%, line 약화와 밴드 위험이 동시에 확정되지 않음"],
      ["청산", "line 약화 + 밴드 하단 위험 또는 밴드 폭 확장"],
      ["확인일", "청산 2일 확인, 재진입 2일 확인"],
      ["포지션", "단일 티커 100% 또는 현금 100%"],
      ["판정", "제품 기본 후보가 아니라 AI 지표 해석용 실험 전략 v1"],
    ],
    visibleFactors: ["conservative", "lowerBand", "bandWidth", "bandExpansion", "bandPercentile"],
  },
];

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, digits)}%`;
}

function formatRatioAsPercent(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return formatPercent(value * 100, digits);
}

function formatUnsignedRatioAsPercent(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${formatNumber(value * 100, digits)}%`;
}

function formatCompact(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function extractErrorMessage(error: unknown) {
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. 127.0.0.1:8000 서버가 켜져 있는지 확인해주세요.";
    }
    return "백테스트 데이터를 불러오는 중 문제가 생겼습니다.";
  }
  return "백테스트 데이터를 불러오지 못했습니다.";
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isValidDate(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && !Number.isNaN(Date.parse(value));
}

function sortUniqueByDate<T extends { date: string }>(rows: T[]) {
  const deduped = new Map<string, T>();
  rows.forEach((row) => {
    if (isValidDate(row.date)) {
      deduped.set(row.date, row);
    }
  });
  return Array.from(deduped.values()).sort((left, right) => left.date.localeCompare(right.date));
}

function sortPriceRows(rows: PriceBar[]) {
  return sortUniqueByDate(rows.filter((row) => Number.isFinite(row.close)));
}

function formatDate(value: Date) {
  return value.toISOString().slice(0, 10);
}

function buildDefaultPriceWindow() {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - DEFAULT_PRICE_LOOKBACK_DAYS);
  return {
    start: formatDate(start),
    end: formatDate(end),
  };
}

async function fetchPriceHistory(ticker: string, timeframe: DisplayTimeframe) {
  const window = buildDefaultPriceWindow();
  const response = await fetchPrices(ticker, {
    timeframe,
    start: window.start,
    end: window.end,
  });
  return sortPriceRows(response.data.data);
}

function getLastFinite(values: number[] | null | undefined) {
  if (!values) {
    return null;
  }
  for (let index = values.length - 1; index >= 0; index -= 1) {
    if (Number.isFinite(values[index])) {
      return values[index];
    }
  }
  return null;
}

function getConservativeValue(prediction: PredictionResult) {
  return getLastFinite(prediction.conservative_series?.length ? prediction.conservative_series : prediction.line_series);
}

function getWorstLowerBandValue(prediction: PredictionResult) {
  const values = prediction.lower_band_series.filter(Number.isFinite);
  return values.length > 0 ? Math.min(...values) : null;
}

function getHighestUpperBandValue(prediction: PredictionResult) {
  const values = prediction.upper_band_series.filter(Number.isFinite);
  return values.length > 0 ? Math.max(...values) : null;
}

function getBandWidthValue(prediction: PredictionResult) {
  const lower = getWorstLowerBandValue(prediction);
  const upper = getHighestUpperBandValue(prediction);
  return lower != null && upper != null ? upper - lower : null;
}

function getStrategyDefinition(strategyId: StrategyId) {
  return STRATEGIES.find((strategy) => strategy.id === strategyId) ?? STRATEGIES[0];
}

function median(values: number[]) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function percentileRank(value: number, values: number[]) {
  const usable = values.filter(Number.isFinite);
  if (usable.length === 0) {
    return null;
  }
  const lowerOrEqual = usable.filter((candidate) => candidate <= value).length;
  return lowerOrEqual / usable.length;
}

function describeAvoidanceStrength(value: number | null | undefined) {
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

function evaluateFollowReturn(result: BacktestSimulationResult | null) {
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

function evaluateDrawdown(result: BacktestSimulationResult | null) {
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

function evaluateLossAvoidance(result: BacktestSimulationResult | null) {
  return describeAvoidanceStrength(result?.largeLossAvoidanceRate);
}

function evaluateTradeFrequency(result: BacktestSimulationResult | null) {
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

function strategyNeedsLine(_: StrategyId) {
  return true;
}

function strategyNeedsBand(_: StrategyId) {
  return true;
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
  conservativeReturn: number;
  lowerBandReturn: number;
  bandWidthExpansion: number | null;
}) {
  const lineGood = row.conservativeReturn >= LENS_BALANCE_RULE.lineEntryThreshold;
  const lineHold = row.conservativeReturn >= LENS_BALANCE_RULE.lineHoldThreshold;
  const lowerRisky = row.lowerBandReturn <= LENS_BALANCE_RULE.lowerRiskThreshold;
  const widthExpanded =
    row.bandWidthExpansion != null && row.bandWidthExpansion >= LENS_BALANCE_RULE.widthExpansionThreshold;
  const entryOk = lineGood && !widthExpanded;
  const riskExit = row.conservativeReturn < LENS_BALANCE_RULE.lineHoldThreshold && (lowerRisky || widthExpanded);

  if (entryOk) {
    return { target: 1 as const, reason: "보수적 예측선이 진입 기준을 충족하고 밴드 폭 급확장이 없습니다." };
  }
  if (lineHold && !riskExit) {
    return { target: 1 as const, reason: "보수적 예측선이 보유 기준 위이고 밴드 위험이 확정되지 않았습니다." };
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
  const lowerRisky = signal.lowerBandReturn <= LENS_BALANCE_RULE.lowerRiskThreshold;
  const widthExpanded =
    signal.bandWidthExpansion != null && signal.bandWidthExpansion >= LENS_BALANCE_RULE.widthExpansionThreshold;
  const lineWeak = signal.conservativeReturn < LENS_BALANCE_RULE.lineHoldThreshold;

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

async function loadStrategySignalCard(ticker: string): Promise<StrategySignalCard> {
  const baseCard: StrategySignalCard = {
    ticker,
    sector: null,
    group: "watch",
    signalLabel: "아직 신호 없음",
    reason: "가격, 예측선, AI 밴드 history 중 일부가 부족해 최신 전략 신호를 만들 수 없습니다.",
    asofDate: null,
    conservativeReturn: null,
    lowerBandReturn: null,
    bandWidthReturn: null,
    bandWidthExpansion: null,
    bandWidthPercentile: null,
    ma60Ratio: null,
    rsi: null,
    hasUsableSignal: false,
  };

  try {
    const [prices, indicatorsResponse, lineResponse, bandResponse, tickerResponse] = await Promise.all([
      fetchPriceHistory(ticker, "1D"),
      fetchIndicators(ticker, { timeframe: "1D", limit: 300 }).catch(() => null),
      fetchPredictionHistory(ticker, { runId: PRODUCT_LINE_RUN_ID, limit: PREDICTION_HISTORY_LIMIT }).catch(() => null),
      fetchPredictionHistory(ticker, { runId: PRODUCT_BAND_RUN_ID, limit: PREDICTION_HISTORY_LIMIT }).catch(() => null),
      fetchTickers({ search: ticker, limit: 1 }).catch(() => null),
    ]);
    const indicators = sortUniqueByDate(indicatorsResponse?.data.data ?? []);
    const lineRows = lineResponse?.data ?? [];
    const bandRows = bandResponse?.data ?? [];
    const sector = tickerResponse?.data.find((item) => item.ticker === ticker)?.sector ?? null;

    if (prices.length === 0 || lineRows.length === 0 || bandRows.length === 0) {
      return { ...baseCard, sector };
    }

    const signals = buildSignals({ priceRows: prices, lineHistory: lineRows, bandHistory: bandRows });
    const latestSignal = signals.at(-1);
    if (!latestSignal) {
      return { ...baseCard, sector };
    }

    const latestIndicator = getLatestIndicatorBefore(indicators, latestSignal.date);
    const classification = classifySignalGroup(latestSignal);
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
      ma60Ratio: latestIndicator?.ma_60_ratio ?? null,
      rsi: latestIndicator?.rsi ?? null,
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
  priceRows: PriceBar[];
  lineHistory: PredictionResult[];
  bandHistory: PredictionResult[];
  feeBps: number;
}): BacktestSimulationResult | null {
  const signals = buildSignals(params);
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
  const strategyId: StrategyId = "lens_balance_v1";
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
    Promise.all(SIGNAL_SCAN_TICKERS.map((ticker) => loadStrategySignalCard(ticker)))
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
  }, []);

  async function loadBacktest(nextTicker: string, nextTimeframe: DisplayTimeframe, nextStrategyId = strategyId) {
    setIsLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    setResult(null);
    setLineHistory([]);
    setBandHistory([]);

    try {
      const normalizedTicker = nextTicker.trim().toUpperCase() || "AAPL";
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

      const [lineResponse, bandResponse] = await Promise.all([
        fetchPredictionHistory(normalizedTicker, {
          runId: PRODUCT_LINE_RUN_ID,
          limit: PREDICTION_HISTORY_LIMIT,
        }).catch(() => null),
        fetchPredictionHistory(normalizedTicker, {
          runId: PRODUCT_BAND_RUN_ID,
          limit: PREDICTION_HISTORY_LIMIT,
        }).catch(() => null),
      ]);
      const lineRows = lineResponse?.data ?? [];
      const bandRows = bandResponse?.data ?? [];
      setLineHistory(lineRows);
      setBandHistory(bandRows);

      const needsLine = strategyNeedsLine(nextStrategyId);
      const needsBand = strategyNeedsBand(nextStrategyId);
      if ((needsLine && lineRows.length === 0) || (needsBand && bandRows.length === 0)) {
        const missingParts = [
          needsLine && lineRows.length === 0 ? "보수적 예측선" : null,
          needsBand && bandRows.length === 0 ? "AI 밴드" : null,
        ].filter(Boolean);
        setStatusMessage(`이 티커에는 아직 ${missingParts.join(", ")} 저장 결과가 없습니다. 가격 데이터만 표시합니다.`);
        return;
      }

      const nextResult = runStrategyBacktest({
        priceRows: prices,
        lineHistory: lineRows,
        bandHistory: bandRows,
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
    void loadBacktest("AAPL", "1D", "lens_balance_v1");
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
          label: "보수적 예측선",
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
      ].filter((factor) => strategyDefinition.visibleFactors.includes(factor.id as StrategyDefinition["visibleFactors"][number]))
    : [];

  return (
    <div className="view-stack">
      <header className="view-header">
        <div className="view-header__title">
          <div className="eyebrow">백테스트</div>
          <h1>전략 신호</h1>
          <p>Lens Balance v1이 보수적 예측선과 AI 밴드를 어떻게 해석했는지 먼저 보고, 종목을 선택해 단일 티커 백테스트를 확인합니다.</p>
        </div>
        <div className="status-badge status-badge--neutral">실험 전략 v1</div>
      </header>

      <section className="panel strategy-signal-overview">
        <div>
          <div className="eyebrow">오늘의 신호 요약</div>
          <h2>Lens Balance v1 AI 지표 기준 신호</h2>
          <p>
            저장된 1D 제품 예측선과 AI 밴드 history가 있는 종목만 읽어 최신 전략 상태를 요약합니다. 포트폴리오 추천이 아니라 단일 티커
            백테스트로 이어지는 신호 탐색 화면입니다.
          </p>
          <p className="strategy-signal-scope-note">현재는 12개 주요 종목 기준 전략 신호입니다. 전체 scanner API가 붙기 전까지는 subset 신호로만 봅니다.</p>
        </div>
        <div className="strategy-signal-summary">
          <div>
            <span>스캔 범위</span>
            <strong>{SIGNAL_SCAN_TICKERS.length}개</strong>
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
          <div className="eyebrow">직접 종목 확인</div>
          <h2>신호 목록에 없는 종목도 확인</h2>
          <p>신호 목록에 없는 종목도 직접 확인할 수 있습니다.</p>
        </div>
        <form className="backtest-toolbar backtest-toolbar--compact" onSubmit={handleSubmit}>
          <label className="search-field">
            <span>티커</span>
            <input value={tickerInput} onChange={(event) => setTickerInput(event.target.value.toUpperCase())} />
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
            이 화면은 매수·매도 추천이 아니라 AI 지표를 해석해 본 룰 기반 시뮬레이션입니다. 현금 보유는 하방 위험과 불확실성이 커졌을 때
            시장 참여를 잠시 줄이는 방식입니다.
          </p>
          <p className="compact-note compact-note--warning">
            검증 판정: 제품 기본 후보가 아니라 실험 전략 v1입니다. 대표 티커별 일관성이 약했던 한계를 숨기지 않고 표시합니다.
          </p>
          <div className="strategy-validation-strip" aria-label="Lens Balance v1 검증 결과">
            {LENS_BALANCE_VALIDATION.map(([label, value]) => (
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
