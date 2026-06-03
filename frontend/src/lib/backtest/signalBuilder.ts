// 전략 신호 빌더 + 카드 분류 헬퍼.
// 부작용 없는 순수 함수. simulationEngine / signalCardLoader / BacktestView JSX가 사용.

import type { IndicatorPoint, PredictionResult, PriceBar } from "@/api/client";
import { INDICATOR_BASELINE_RULE, LENS_BALANCE_RULE } from "@/lib/backtest/constants";
import {
  getBandWidthValue,
  getConservativeValue,
  getWorstLowerBandValue,
  median,
  percentileRank,
} from "@/lib/backtest/predictionHelpers";
import type { RiskSignal, StrategySignalCard } from "@/lib/backtest/types";
import { isFiniteNumber } from "@/lib/formatters";

export function buildRawSignals(params: {
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

export function getRawTarget(row: {
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

export function normalizeRsi(value: number | null | undefined) {
  if (!Number.isFinite(value ?? NaN)) {
    return null;
  }
  const number = Number(value);
  return number >= 0 && number <= 1 ? number * 100 : number;
}

export function buildIndicatorRows(priceRows: PriceBar[], indicators: IndicatorPoint[]) {
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

export function getIndicatorRawTarget(row: ReturnType<typeof buildIndicatorRows>[number]) {
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

export function buildIndicatorSignals(params: { priceRows: PriceBar[]; indicators: IndicatorPoint[] }) {
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

export function buildSignals(params: {
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

export function getLatestIndicatorBefore(indicators: IndicatorPoint[], date: string | null) {
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

export function getBandWidthState(card: Pick<StrategySignalCard, "bandWidthExpansion" | "bandWidthPercentile" | "bandWidthReturn">) {
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

export function classifySignalGroup(signal: RiskSignal) {
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

export function classifyIndicatorSignal(signal: RiskSignal) {
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
