import type { IndicatorPoint } from "@/api/client";
import type { IndicatorChartPoint } from "@/components/IndicatorPanel";

import { sortUniqueByDate } from "./dateUtils";
import { formatNumber, formatSignedPercentPoint, isFiniteNumber, normalizeRsi, ratioToPercent } from "./formatters";

export type IndicatorId =
  | "rsi"
  | "macd_ratio"
  | "vol_change"
  | "ma_5_ratio"
  | "ma_20_ratio"
  | "ma_60_ratio"
  | "bb_position"
  | "atr_ratio"
  | "ai_band_width";

export interface IndicatorDefinition {
  id: IndicatorId;
  label: string;
  field?: keyof IndicatorPoint;
  category: "기본" | "추세" | "변동성/위치" | "AI";
  description: string;
  color: string;
  baseline?: number;
  fixedRange?: {
    min: number;
    max: number;
  };
  transform?: (value: number) => number;
  formatLatest: (value: number | null | undefined) => string;
}

export const DEFAULT_INDICATORS: IndicatorId[] = ["rsi", "macd_ratio", "ai_band_width"];

export const INDICATOR_DEFINITIONS: IndicatorDefinition[] = [
  {
    id: "rsi",
    label: "RSI",
    field: "rsi",
    category: "기본",
    description: "최근 상승/하락 강도를 0~100으로 본 과열·침체 참고 지표",
    color: "#0f766e",
    fixedRange: { min: 0, max: 100 },
    transform: normalizeRsi,
    formatLatest: (value) => formatNumber(value, 1),
  },
  {
    id: "macd_ratio",
    label: "MACD",
    field: "macd_ratio",
    category: "기본",
    description: "추세 전환과 모멘텀 변화를 보는 이동평균 기반 지표",
    color: "#2563eb",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "vol_change",
    label: "거래량 변화",
    field: "vol_change",
    category: "기본",
    description: "직전 구간 대비 거래량 증감을 봅니다.",
    color: "#64748b",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ma_5_ratio",
    label: "MA 5 괴리",
    field: "ma_5_ratio",
    category: "추세",
    description: "5일 이동평균 대비 가격 위치를 봅니다.",
    color: "#7c3aed",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ma_20_ratio",
    label: "MA 20 괴리",
    field: "ma_20_ratio",
    category: "추세",
    description: "20일 이동평균 대비 가격 위치를 봅니다.",
    color: "#7c3aed",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ma_60_ratio",
    label: "MA 60 괴리",
    field: "ma_60_ratio",
    category: "추세",
    description: "60일 이동평균 대비 중기 위치를 봅니다.",
    color: "#7c3aed",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "bb_position",
    label: "BB 위치",
    field: "bb_position",
    category: "변동성/위치",
    description: "볼린저 밴드 안에서 현재 위치를 봅니다.",
    color: "#9333ea",
    fixedRange: { min: 0, max: 1 },
    formatLatest: (value) => formatNumber(value, 2),
  },
  {
    id: "atr_ratio",
    label: "ATR",
    field: "atr_ratio",
    category: "변동성/위치",
    description: "최근 가격 변동 폭을 보는 전통 변동성 지표",
    color: "#d97706",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ai_band_width",
    label: "AI 밴드 폭",
    category: "AI",
    description: "AI가 보는 예상 변동 범위의 넓이",
    color: "#4b5563",
    baseline: 0,
    formatLatest: (value) => formatNumber(value, 4),
  },
];

export function getIndicatorValue(row: IndicatorPoint, definition: IndicatorDefinition) {
  if (!definition.field) {
    return null;
  }
  const rawValue = row[definition.field];
  if (!isFiniteNumber(rawValue)) {
    return null;
  }
  return definition.transform ? definition.transform(rawValue) : rawValue;
}

export function hasIndicatorValues(rows: IndicatorPoint[], definition: IndicatorDefinition) {
  if (!definition.field) {
    return false;
  }
  return rows.some((row) => isFiniteNumber(getIndicatorValue(row, definition)));
}

export function buildFieldPoints(
  rows: IndicatorPoint[],
  definition: IndicatorDefinition,
  allowedDates?: Set<string>
): IndicatorChartPoint[] {
  return sortUniqueByDate(
    rows
      .map((row) => {
        if (allowedDates && !allowedDates.has(row.date)) {
          return null;
        }
        const value = getIndicatorValue(row, definition);
        if (!isFiniteNumber(value)) {
          return null;
        }
        return {
          date: row.date,
          value,
        };
      })
      .filter((point): point is IndicatorChartPoint => point !== null)
  );
}
