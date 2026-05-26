"use client";

import { FormEvent, startTransition, useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";

import {
  DisplayTimeframe,
  EvaluationSummary,
  fetchIndicators,
  fetchPrices,
  fetchTickers,
  fetchV1Band1dPrediction,
  fetchV1Band1wPrediction,
  fetchV1LinePrediction,
  getBackendConfigWarning,
  IndicatorPoint,
  isPredictionTimeframeEnabled,
  PredictionResult,
  PriceBar,
  ProductBandHistoryPoint,
  ProductLineHistoryPoint,
  StockSummary,
  V1BandPredictionResult,
  V1LinePredictionResult,
} from "@/api/client";
import Chart from "@/components/Chart";
import IndicatorPanel, { IndicatorChartPoint, IndicatorChartSeries } from "@/components/IndicatorPanel";
import LayerToggle from "@/components/LayerToggle";
import { getProductBandSlot, getProductLineSlot, PRODUCT_SLOT_BY_ID, ProductSlot } from "@/lib/productSlots";

type ChartType = "candles" | "line";
type IndicatorId =
  | "rsi"
  | "macd_ratio"
  | "vol_change"
  | "ma_5_ratio"
  | "ma_20_ratio"
  | "ma_60_ratio"
  | "bb_position"
  | "atr_ratio"
  | "ai_band_width";

interface AiState {
  kind: "loading" | "ready" | "disabled" | "empty";
  message: string;
}

interface PriceState {
  kind: "loading" | "ready" | "empty" | "error";
  message: string;
}

interface ApiErrorShape {
  code: string;
  message: string;
}

interface IndicatorDefinition {
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

interface PredictionOverlayCheck {
  ok: boolean;
  message: string | null;
}

interface PredictionProvenance {
  latestRunId: string | null;
  selectedRunId: string | null;
  isFallback: boolean;
}

const TIMEFRAME_OPTIONS: Array<{ value: DisplayTimeframe; label: string }> = [
  { value: "1D", label: "1D" },
  { value: "1W", label: "1W" },
  { value: "1M", label: "1M" },
];

const DEFAULT_INDICATORS: IndicatorId[] = ["rsi", "macd_ratio", "ai_band_width"];
const PRODUCT_HISTORY_LOOKBACK_DAYS = 370;
const DEFAULT_PRICE_LOOKBACK_DAYS = 365;
const FULL_PRICE_HISTORY_START_YEAR = 2015;
const LINE_OVERLAY_HELD = false;

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatVolume(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}%`;
}

function formatRatio(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  if (Math.abs(value) <= 1) {
    return `${formatNumber(value * 100, 1)}%`;
  }
  return formatNumber(value, 2);
}

function formatSignedPercentPoint(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}%`;
}

function getMetaString(meta: Record<string, unknown> | undefined, key: string) {
  const value = meta?.[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function normalizeRsi(value: number) {
  return value >= 0 && value <= 1 ? value * 100 : value;
}

function ratioToPercent(value: number) {
  return value * 100;
}

const INDICATOR_DEFINITIONS: IndicatorDefinition[] = [
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

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function extractApiError(error: unknown): ApiErrorShape | null {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const payload = (error as { response?: { data?: { error?: { code?: string; message?: string } } } }).response?.data
      ?.error;
    if (payload?.code && payload?.message) {
      return {
        code: payload.code,
        message: payload.message,
      };
    }
  }
  return null;
}

function extractHttpStatus(error: unknown) {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const status = (error as { response?: { status?: unknown } }).response?.status;
    return typeof status === "number" ? status : null;
  }
  return null;
}

function isResourceNotFound(error: unknown) {
  const apiError = extractApiError(error);
  return apiError?.code === "RESOURCE_NOT_FOUND" || extractHttpStatus(error) === 404;
}

function isBackendConnectionError(error: unknown) {
  const status = extractHttpStatus(error);
  if (status != null) {
    return status >= 500;
  }
  if (error instanceof Error) {
    return (
      error.message === "Network Error" ||
      error.message.includes("ECONNREFUSED") ||
      error.message.includes("Failed to construct 'URL'") ||
      error.message.includes("Invalid URL")
    );
  }
  return false;
}

function extractErrorMessage(error: unknown) {
  const apiError = extractApiError(error);
  if (apiError) {
    return apiError.message;
  }
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 설정과 백엔드 상태를 확인해주세요.";
    }
    if (error.message.includes("Failed to construct 'URL'") || error.message.includes("Invalid URL")) {
      return "백엔드 주소 설정을 확인할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 값을 확인해주세요.";
    }
    return error.message;
  }
  return "데이터를 불러오지 못했습니다.";
}

function buildPriceMissingMessage(ticker: string, timeframe: DisplayTimeframe) {
  return `${ticker.toUpperCase()} ${timeframe} 가격 데이터가 아직 연결되지 않았습니다. 티커 또는 데이터 소스를 확인해주세요.`;
}

function getLastPrice(rows: PriceBar[]) {
  return rows.length > 0 ? rows[rows.length - 1] : null;
}

function getChangePercent(rows: PriceBar[]) {
  if (rows.length < 2) {
    return null;
  }
  const latest = rows[rows.length - 1].close;
  const previous = rows[rows.length - 2].close;
  if (!previous) {
    return null;
  }
  return ((latest - previous) / previous) * 100;
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

function getLastPoint(points: IndicatorChartPoint[]) {
  return points.length > 0 ? points[points.length - 1] : null;
}

function buildAiState(timeframe: DisplayTimeframe): AiState {
  if (timeframe === "1M") {
    return {
      kind: "disabled",
      message: "월간 화면은 현재 가격 전용입니다.",
    };
  }
  return {
    kind: "empty",
    message: "저장된 최신 예측 결과가 아직 없습니다.",
  };
}

type SlotFreshness = "fresh" | "delayed" | "stale" | "static" | "deferred" | "empty";

function diffCalendarDaysBetween(fromDate: string | null | undefined, toDate: string | null | undefined) {
  if (!isValidDate(fromDate) || !isValidDate(toDate)) {
    return null;
  }
  const from = new Date(`${fromDate}T00:00:00Z`).getTime();
  const to = new Date(`${toDate}T00:00:00Z`).getTime();
  if (!Number.isFinite(from) || !Number.isFinite(to)) {
    return null;
  }
  return Math.max(0, Math.floor((to - from) / (24 * 60 * 60 * 1000)));
}

function countPriceRowsAfter(rows: PriceBar[], fromDate: string | null | undefined, toDate: string | null | undefined) {
  if (!isValidDate(fromDate) || !isValidDate(toDate)) {
    return null;
  }
  return sortPriceRows(rows).filter((row) => row.date > fromDate && row.date <= toDate).length;
}

function getSlotFreshness(
  slot: ProductSlot | null | undefined,
  asofDate: string | null | undefined,
  latestPriceDate: string | null | undefined,
  priceRows: PriceBar[]
): SlotFreshness {
  if (!slot) {
    return "empty";
  }
  if (slot.refreshPolicy === "deferred") {
    return "deferred";
  }
  if (slot.refreshPolicy === "static") {
    return "static";
  }
  if (!isValidDate(asofDate) || !isValidDate(latestPriceDate)) {
    return "empty";
  }

  if (slot.timeframe === "1D") {
    const tradingRowsAfter = countPriceRowsAfter(priceRows, asofDate, latestPriceDate);
    if (tradingRowsAfter == null) {
      return "empty";
    }
    if (slot.staleAfterDays != null && tradingRowsAfter >= slot.staleAfterDays) {
      return "stale";
    }
    if (slot.freshAfterDays != null && tradingRowsAfter <= slot.freshAfterDays) {
      return "fresh";
    }
    return "delayed";
  }

  const ageDays = diffCalendarDaysBetween(asofDate, latestPriceDate);
  if (ageDays == null) {
    return "empty";
  }
  if (slot.staleAfterDays != null && ageDays >= slot.staleAfterDays) {
    return "stale";
  }
  if (slot.freshAfterDays != null && ageDays <= slot.freshAfterDays) {
    return "fresh";
  }
  return "delayed";
}

function getFutureForecastDates(prediction: PredictionResult | null | undefined, latestPriceDate: string | null | undefined) {
  if (!prediction || !isValidDate(latestPriceDate)) {
    return [];
  }
  return uniqueDates((prediction.forecast_dates ?? []).filter((date) => isValidDate(date) && date > latestPriceDate));
}

function formatFreshnessStatus(status: SlotFreshness) {
  const labels: Record<SlotFreshness, string> = {
    fresh: "fresh",
    delayed: "delayed",
    stale: "stale",
    static: "static",
    deferred: "deferred",
    empty: "데이터 없음",
  };
  return labels[status];
}

function getFreshnessClass(status: SlotFreshness) {
  if (status === "fresh") {
    return "fresh";
  }
  if (status === "stale") {
    return "stale";
  }
  if (status === "static" || status === "deferred") {
    return "static";
  }
  if (status === "delayed") {
    return "delayed";
  }
  return "empty";
}

function getIndicatorValue(row: IndicatorPoint, definition: IndicatorDefinition) {
  if (!definition.field) {
    return null;
  }
  const rawValue = row[definition.field];
  if (!isFiniteNumber(rawValue)) {
    return null;
  }
  return definition.transform ? definition.transform(rawValue) : rawValue;
}

function hasIndicatorValues(rows: IndicatorPoint[], definition: IndicatorDefinition) {
  if (!definition.field) {
    return false;
  }
  return rows.some((row) => isFiniteNumber(getIndicatorValue(row, definition)));
}

function buildFieldPoints(rows: IndicatorPoint[], definition: IndicatorDefinition, allowedDates?: Set<string>): IndicatorChartPoint[] {
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

function isValidDate(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && !Number.isNaN(Date.parse(value));
}

function uniqueDates(dates: string[]) {
  return Array.from(new Set(dates.filter(isValidDate))).sort((left, right) => left.localeCompare(right));
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

function sortUniqueByAsofDate<T extends { asof_date: string }>(rows: T[]) {
  const deduped = new Map<string, T>();
  rows.forEach((row) => {
    if (isValidDate(row.asof_date)) {
      deduped.set(row.asof_date, row);
    }
  });
  return Array.from(deduped.values()).sort((left, right) => left.asof_date.localeCompare(right.asof_date));
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

function buildFullPriceWindows() {
  const currentYear = new Date().getFullYear();
  return Array.from({ length: currentYear - FULL_PRICE_HISTORY_START_YEAR + 1 }, (_, index) => {
    const year = FULL_PRICE_HISTORY_START_YEAR + index;
    return {
      start: `${year}-01-01`,
      end: `${year}-12-31`,
    };
  });
}

function addDays(date: string, days: number) {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return date;
  }
  parsed.setUTCDate(parsed.getUTCDate() + days);
  return parsed.toISOString().slice(0, 10);
}

function addBusinessDays(date: string, days: number) {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return date;
  }

  let remaining = days;
  while (remaining > 0) {
    parsed.setUTCDate(parsed.getUTCDate() + 1);
    const day = parsed.getUTCDay();
    if (day !== 0 && day !== 6) {
      remaining -= 1;
    }
  }
  return parsed.toISOString().slice(0, 10);
}

function finiteOrNull(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeBandValueToPrice(value: number, baseClose: number | null) {
  if (Math.abs(value) < 1 && baseClose != null) {
    return baseClose * (1 + value);
  }
  if (Math.abs(value) < 1) {
    return null;
  }
  return value;
}

function findCloseOnOrBefore(rows: PriceBar[], date: string) {
  let selected: number | null = null;
  for (const row of rows) {
    if (!isValidDate(row.date) || row.date > date) {
      continue;
    }
    if (Number.isFinite(row.close)) {
      selected = row.close;
    }
  }
  return selected;
}

function findTradingHorizonDate(rows: PriceBar[], asofDate: string, horizon: number) {
  const dates = sortPriceRows(rows).map((row) => row.date);
  let baseIndex = -1;
  for (let index = 0; index < dates.length; index += 1) {
    if (dates[index] <= asofDate) {
      baseIndex = index;
    } else {
      break;
    }
  }

  if (baseIndex >= 0) {
    const targetIndex = baseIndex + horizon;
    if (targetIndex < dates.length) {
      return dates[targetIndex];
    }
  }

  return addBusinessDays(asofDate, horizon);
}

function buildLineHistoryFromV1(response: V1LinePredictionResult, priceRows: PriceBar[]): ProductLineHistoryPoint[] {
  const slot = PRODUCT_SLOT_BY_ID["line-1d"];
  const rows: ProductLineHistoryPoint[] = [];
  response.data.forEach((row) => {
    const returnValue = finiteOrNull(row.safe_line_score) ?? finiteOrNull(row.line_score);
    const baseClose = findCloseOnOrBefore(priceRows, row.asof_date);
    if (!isValidDate(row.asof_date) || returnValue == null || baseClose == null) {
      return;
    }
    rows.push({
      asof_date: row.asof_date,
      forecast_date: findTradingHorizonDate(priceRows, row.asof_date, slot.horizon ?? 5),
      display_horizon: slot.horizon ?? 5,
      value: baseClose * (1 + returnValue),
      run_id: row.model_id ?? slot.runId ?? slot.sourceCp ?? "line-serving",
    });
  });
  return sortUniqueByAsofDate(rows);
}

function buildLinePredictionFromV1(response: V1LinePredictionResult, priceRows: PriceBar[]): PredictionResult | null {
  const slot = PRODUCT_SLOT_BY_ID["line-1d"];
  const sortedRows = [...response.data].filter((row) => isValidDate(row.asof_date)).sort((left, right) => left.asof_date.localeCompare(right.asof_date));
  const latest = sortedRows.at(-1);
  if (!latest) {
    return null;
  }

  const returnValue = finiteOrNull(latest.safe_line_score) ?? finiteOrNull(latest.line_score);
  const baseClose = findCloseOnOrBefore(priceRows, latest.asof_date);
  if (returnValue == null || baseClose == null) {
    return null;
  }

  const horizon = slot.horizon ?? 5;
  const lineValue = baseClose * (1 + returnValue);
  return {
    ticker: response.ticker,
    model_name: slot.modelName,
    timeframe: "1D",
    horizon,
    asof_date: latest.asof_date,
    decision_time: latest.asof_date,
    run_id: latest.model_id ?? response.model_id ?? slot.runId ?? slot.sourceCp ?? "line-serving",
    model_ver: slot.version ?? "v1",
    signal: "HOLD",
    forecast_dates: [findTradingHorizonDate(priceRows, latest.asof_date, horizon)],
    upper_band_series: [],
    lower_band_series: [],
    conservative_series: [lineValue],
    line_series: [lineValue],
    band_quantile_low: null,
    band_quantile_high: null,
    meta: {
      role: "line_model",
      layer: "line",
      source_cp: latest.source_cp ?? response.source_cp ?? slot.sourceCp,
      model_id: latest.model_id ?? response.model_id ?? slot.modelId,
      value_unit: "price",
      raw_value_unit: "return",
      transform: "asof_close_times_one_plus_h5_safe_line_score",
      display_horizon: "h5",
      refresh_policy: "active_serving_parquet",
    },
  };
}

function selectBandDisplayRows(response: V1BandPredictionResult, preferredHorizon: number) {
  const byAsof = new Map<string, V1BandPredictionResult["data"]>();
  response.data.forEach((row) => {
    if (!isValidDate(row.asof_date)) {
      return;
    }
    const rows = byAsof.get(row.asof_date) ?? [];
    rows.push(row);
    byAsof.set(row.asof_date, rows);
  });

  return Array.from(byAsof.entries())
    .map(([asofDate, rows]) => {
      const preferred = rows.find((row) => row.horizon_step === preferredHorizon);
      const fallback = rows.sort((left, right) => right.horizon_step - left.horizon_step)[0];
      return preferred ?? fallback ?? null;
    })
    .filter((row): row is V1BandPredictionResult["data"][number] => row !== null);
}

function buildBandHistoryFromV1(
  response: V1BandPredictionResult,
  preferredHorizon: number,
  priceRows: PriceBar[]
): ProductBandHistoryPoint[] {
  return sortUniqueByAsofDate(
    selectBandDisplayRows(response, preferredHorizon)
      .map((row) => {
        const rawLower = finiteOrNull(row.band_lower);
        const rawUpper = finiteOrNull(row.band_upper);
        const baseClose = findCloseOnOrBefore(priceRows, row.asof_date);
        const lower = rawLower == null ? null : normalizeBandValueToPrice(rawLower, baseClose);
        const upper = rawUpper == null ? null : normalizeBandValueToPrice(rawUpper, baseClose);
        if (!isValidDate(row.asof_date) || lower == null || upper == null) {
          return null;
        }
        return {
          asof_date: row.asof_date,
          display_horizon: row.horizon_step,
          lower,
          upper,
          run_id: row.model_id ?? response.model_id ?? response.source_cp ?? response.slot,
        };
      })
      .filter((row): row is ProductBandHistoryPoint => row !== null)
  );
}

function buildBandPredictionFromV1(
  response: V1BandPredictionResult,
  timeframe: "1D" | "1W",
  preferredHorizon: number,
  priceRows: PriceBar[]
): PredictionResult | null {
  const sortedRows = [...response.data].filter((row) => isValidDate(row.asof_date)).sort((left, right) => {
    const byDate = left.asof_date.localeCompare(right.asof_date);
    return byDate !== 0 ? byDate : left.horizon_step - right.horizon_step;
  });
  const latestAsof = sortedRows.at(-1)?.asof_date;
  if (!latestAsof) {
    return null;
  }
  const latestRows = sortedRows
    .filter((row) => row.asof_date === latestAsof)
    .sort((left, right) => left.horizon_step - right.horizon_step);
  const rows = latestRows.filter((row) => row.horizon_step <= preferredHorizon);
  const forecastDates = rows.map((row) => row.forecast_date ?? addDays(row.asof_date, timeframe === "1W" ? row.horizon_step * 7 : row.horizon_step));
  const baseClose = findCloseOnOrBefore(priceRows, latestAsof);
  const upper = rows.map((row) => {
    const value = finiteOrNull(row.band_upper);
    return value == null ? null : normalizeBandValueToPrice(value, baseClose);
  });
  const lower = rows.map((row) => {
    const value = finiteOrNull(row.band_lower);
    return value == null ? null : normalizeBandValueToPrice(value, baseClose);
  });
  if (forecastDates.length === 0 || upper.some((value) => value == null) || lower.some((value) => value == null)) {
    return null;
  }
  const slot = timeframe === "1W" ? PRODUCT_SLOT_BY_ID["band-1w"] : PRODUCT_SLOT_BY_ID["band-1d"];
  return {
    ticker: response.ticker,
    model_name: slot.modelName,
    timeframe,
    horizon: preferredHorizon,
    asof_date: latestAsof,
    decision_time: latestAsof,
    run_id: response.model_id ?? slot.runId ?? slot.sourceCp ?? response.slot,
    model_ver: slot.version ?? "v1",
    signal: "HOLD",
    forecast_dates: forecastDates,
    upper_band_series: upper as number[],
    lower_band_series: lower as number[],
    conservative_series: [],
    line_series: [],
    band_quantile_low: null,
    band_quantile_high: null,
    meta: {
      role: "band_model",
      layer: "band",
      source_cp: response.source_cp ?? slot.sourceCp,
      model_id: response.model_id ?? slot.modelId,
      value_unit: "price",
      raw_value_unit: baseClose == null ? "price" : "auto_normalized_to_price",
      refresh_policy: "auto_parquet",
    },
  };
}

async function fetchPriceHistory(ticker: string, timeframe: DisplayTimeframe, fullHistory = false) {
  if (!fullHistory) {
    const window = buildDefaultPriceWindow();
    const response = await fetchPrices(ticker, {
      timeframe,
      start: window.start,
      end: window.end,
    });
    return sortPriceRows(response.data.data);
  }

  const responses = await Promise.all(
    buildFullPriceWindows().map((window) =>
      fetchPrices(ticker, {
        timeframe,
        start: window.start,
        end: window.end,
      })
    )
  );
  return sortPriceRows(responses.flatMap((response) => response.data.data));
}

function isLegacyPrediction(prediction: PredictionResult | null) {
  if (!prediction) {
    return false;
  }
  const role = getMetaString(prediction.meta, "role");
  const deprecated = prediction.meta?.deprecated_for_phase1_product_contract;
  return (
    prediction.model_name.includes("composite") ||
    role === "composite_model" ||
    deprecated === true ||
    deprecated === "true"
  );
}

function checkLineOverlay(prediction: PredictionResult): PredictionOverlayCheck {
  const forecastDates = prediction.forecast_dates ?? [];
  const lineValues = prediction.conservative_series?.length ? prediction.conservative_series : prediction.line_series ?? [];

  if (forecastDates.length === 0) {
    return { ok: false, message: "예측 날짜가 없어 보수적 기준선을 숨겼습니다." };
  }
  if (
    lineValues.length !== forecastDates.length
  ) {
    return { ok: false, message: "예측 날짜와 시리즈 길이가 맞지 않아 보수적 기준선을 숨겼습니다." };
  }

  const allValues = [...lineValues];
  if (forecastDates.some((date) => !date) || allValues.some((value) => !Number.isFinite(value))) {
    return { ok: false, message: "예측 데이터에 표시할 수 없는 값이 있어 보수적 기준선을 숨겼습니다." };
  }

  return { ok: true, message: null };
}

function checkBandOverlay(prediction: PredictionResult): PredictionOverlayCheck {
  const forecastDates = prediction.forecast_dates ?? [];

  if (forecastDates.length === 0) {
    return { ok: false, message: "예측 날짜가 없어 AI 밴드를 숨겼습니다." };
  }
  if (
    prediction.upper_band_series.length !== forecastDates.length ||
    prediction.lower_band_series.length !== forecastDates.length
  ) {
    return { ok: false, message: "예측 날짜와 밴드 길이가 맞지 않아 AI 밴드를 숨겼습니다." };
  }

  const allValues = [...prediction.upper_band_series, ...prediction.lower_band_series];
  if (forecastDates.some((date) => !date) || allValues.some((value) => !Number.isFinite(value))) {
    return { ok: false, message: "AI 밴드 데이터에 표시할 수 없는 값이 있어 숨겼습니다." };
  }

  return { ok: true, message: null };
}

function isActualBandPrediction(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return false;
  }

  const role = getMetaString(prediction.meta, "role");
  const bandFieldsPolicy = getMetaString(prediction.meta, "band_fields_policy");
  const bandSavedInCp140 = prediction.meta?.band_saved_in_cp140;

  if (role !== "band_model") {
    return false;
  }

  if (bandSavedInCp140 === false || bandSavedInCp140 === "false") {
    return false;
  }

  return bandFieldsPolicy !== "schema_required_degenerate_equal_to_line";
}

function getPredictionLineValues(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return [];
  }
  return prediction.conservative_series?.length ? prediction.conservative_series : prediction.line_series ?? [];
}

function isPredictionLineWithinPriceRange(prediction: PredictionResult, rows: PriceBar[]) {
  const values = getPredictionLineValues(prediction).filter(Number.isFinite);
  const priceValues = rows
    .flatMap((row) => [row.low ?? row.close, row.high ?? row.close, row.close])
    .filter(Number.isFinite);
  if (values.length === 0 || priceValues.length < 2) {
    return false;
  }

  const minPrice = Math.min(...priceValues);
  const maxPrice = Math.max(...priceValues);
  const latestPrice = priceValues[priceValues.length - 1] ?? maxPrice;
  const span = Math.max(maxPrice - minPrice, Math.abs(latestPrice) * 0.04, 1);
  const lowerBound = minPrice - span * 0.35;
  const upperBound = maxPrice + span * 0.35;
  return values.every((value) => value >= lowerBound && value <= upperBound);
}

function buildAiBandWidthPoints(history: ProductBandHistoryPoint[], latestPrediction: PredictionResult | null): IndicatorChartPoint[] {
  const historyPoints = history
    .map((row) => {
      if (!row.asof_date || !Number.isFinite(row.upper) || !Number.isFinite(row.lower)) {
        return null;
      }
      return {
        date: row.asof_date,
        value: Math.max(0, row.upper - row.lower),
      };
    })
    .filter((point): point is IndicatorChartPoint => point !== null);

  const latestUpper = latestPrediction?.upper_band_series?.[0];
  const latestLower = latestPrediction?.lower_band_series?.[0];
  if (latestPrediction?.asof_date && isFiniteNumber(latestUpper) && isFiniteNumber(latestLower)) {
    historyPoints.push({
      date: latestPrediction.asof_date,
      value: Math.max(0, latestUpper - latestLower),
    });
  }

  return sortUniqueByDate(historyPoints);
}

function buildVisibleIndicatorSeries(params: {
  selectedIndicators: IndicatorId[];
  indicatorData: IndicatorPoint[];
  indicatorErrorMessage: string | null;
  availableDefinitions: IndicatorDefinition[];
  allowedDates?: Set<string>;
  aiBandWidthPoints: IndicatorChartPoint[];
}): IndicatorChartSeries[] {
  const { selectedIndicators, indicatorData, indicatorErrorMessage, availableDefinitions, allowedDates, aiBandWidthPoints } = params;
  const indicatorEmptyMessage = indicatorErrorMessage ?? "저장된 보조지표 데이터가 없습니다.";

  return availableDefinitions
    .filter((definition) => selectedIndicators.includes(definition.id))
    .map((definition) => {
      if (definition.id === "ai_band_width") {
        const latest = getLastPoint(aiBandWidthPoints)?.value;
        return {
          id: definition.id,
          label: definition.label,
          groupLabel: definition.category,
          points: aiBandWidthPoints,
          color: definition.color,
          baseline: definition.baseline,
          fixedRange: definition.fixedRange,
          latestLabel: definition.formatLatest(latest),
          emptyMessage: "저장된 AI 밴드 폭 이력이 없습니다.",
        };
      }

      const points = buildFieldPoints(indicatorData, definition, allowedDates);
      const latest = getLastPoint(points)?.value;

      return {
        id: definition.id,
        label: definition.label,
        groupLabel: definition.category,
        points,
        color: definition.color,
        baseline: definition.baseline,
        fixedRange: definition.fixedRange,
        latestLabel: definition.formatLatest(latest),
        emptyMessage: indicatorEmptyMessage,
      };
    });
}

export default function StockView() {
  const [tickerInput, setTickerInput] = useState("AAPL");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [timeframe, setTimeframe] = useState<DisplayTimeframe>("1D");
  const [chartType, setChartType] = useState<ChartType>("candles");
  const [layers, setLayers] = useState({
    aiBand: true,
    conservativeLine: true,
    volumeBar: true,
  });
  const [selectedIndicators, setSelectedIndicators] = useState<IndicatorId[]>(DEFAULT_INDICATORS);
  const [priceData, setPriceData] = useState<PriceBar[]>([]);
  const [indicatorData, setIndicatorData] = useState<IndicatorPoint[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [bandPrediction, setBandPrediction] = useState<PredictionResult | null>(null);
  const [linePredictionHistory, setLinePredictionHistory] = useState<ProductLineHistoryPoint[]>([]);
  const [bandPredictionHistory, setBandPredictionHistory] = useState<ProductBandHistoryPoint[]>([]);
  const [predictionProvenance, setPredictionProvenance] = useState<PredictionProvenance>({
    latestRunId: null,
    selectedRunId: null,
    isFallback: false,
  });
  const [bandEvaluation, setBandEvaluation] = useState<EvaluationSummary | null>(null);
  const [aiState, setAiState] = useState<AiState>({ kind: "loading", message: "예측 확인 중" });
  const [priceState, setPriceState] = useState<PriceState>({ kind: "loading", message: "가격 확인 중" });
  const [suggestions, setSuggestions] = useState<StockSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchErrorMessage, setSearchErrorMessage] = useState<string | null>(null);
  const [indicatorErrorMessage, setIndicatorErrorMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [visibleTimelineDates, setVisibleTimelineDates] = useState<string[]>([]);
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
          setSearchErrorMessage(null);
        }
      })
      .catch(() => {
        if (active) {
          setSuggestions([]);
          setSearchErrorMessage("티커 검색을 사용할 수 없습니다. 티커를 직접 입력하면 가격 조회는 계속 가능합니다.");
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

  async function loadStock(nextTicker: string, nextTimeframe: DisplayTimeframe) {
    setIsLoading(true);
    setErrorMessage(null);
    setIndicatorErrorMessage(null);
    setAiState({ kind: "loading", message: "예측 확인 중" });
    setPriceState({ kind: "loading", message: "가격 확인 중" });
    setPrediction(null);
    setBandPrediction(null);
    setLinePredictionHistory([]);
    setBandPredictionHistory([]);
    setBandEvaluation(null);
    setPredictionProvenance({
      latestRunId: null,
      selectedRunId: null,
      isFallback: false,
    });

    try {
      let priceRows: PriceBar[] = [];
      try {
        priceRows = await fetchPriceHistory(nextTicker, nextTimeframe);
        setPriceData(priceRows);
        setPriceState(
          priceRows.length > 0
            ? { kind: "ready", message: "가격 데이터가 연결되었습니다." }
            : { kind: "empty", message: buildPriceMissingMessage(nextTicker, nextTimeframe) }
        );
      } catch (priceError) {
        priceRows = [];
        setPriceData([]);
        if (isResourceNotFound(priceError)) {
          setPriceState({ kind: "empty", message: buildPriceMissingMessage(nextTicker, nextTimeframe) });
        } else {
          const safeMessage = extractErrorMessage(priceError);
          setPriceState({ kind: "error", message: safeMessage });
          if (isBackendConnectionError(priceError)) {
            setErrorMessage(safeMessage);
          }
        }
      }

      try {
        const indicatorsResponse = await fetchIndicators(nextTicker, { timeframe: nextTimeframe, limit: 300 });
        setIndicatorData(sortUniqueByDate(indicatorsResponse.data.data));
        setIndicatorErrorMessage(null);
      } catch {
        setIndicatorData([]);
        setIndicatorErrorMessage("보조지표를 불러오지 못했습니다.");
      }

      if (!isPredictionTimeframeEnabled(nextTimeframe)) {
        setPrediction(null);
        setLinePredictionHistory([]);
        setBandPredictionHistory([]);
        setPredictionProvenance({
          latestRunId: null,
          selectedRunId: null,
          isFallback: false,
        });
        setAiState(
          priceRows.length === 0
            ? { kind: "disabled", message: "월간 화면은 가격 데이터가 연결되면 가격 전용 차트로 표시됩니다." }
            : buildAiState(nextTimeframe)
        );
        return;
      }

      const lineSlot = getProductLineSlot(nextTimeframe);
      const bandSlot = getProductBandSlot(nextTimeframe);
      const linePromise =
        !LINE_OVERLAY_HELD && nextTimeframe === "1D" && lineSlot?.status === "active"
          ? fetchV1LinePrediction(nextTicker, { days: PRODUCT_HISTORY_LOOKBACK_DAYS }).catch(() => null)
          : Promise.resolve(null);
      const bandPromise =
        bandSlot?.status === "active"
          ? nextTimeframe === "1W"
            ? fetchV1Band1wPrediction(nextTicker, { days: 730 }).catch(() => null)
            : fetchV1Band1dPrediction(nextTicker, { days: PRODUCT_HISTORY_LOOKBACK_DAYS }).catch(() => null)
          : Promise.resolve(null);

      const [lineResponse, bandResponse] = await Promise.all([linePromise, bandPromise]);
      const lineHistory = lineResponse ? buildLineHistoryFromV1(lineResponse.data, priceRows) : [];
      const lineLatest = lineResponse ? buildLinePredictionFromV1(lineResponse.data, priceRows) : null;
      const bandHistory = bandResponse ? buildBandHistoryFromV1(bandResponse.data, bandSlot?.horizon ?? 5, priceRows) : [];
      const bandLatest = bandResponse ? buildBandPredictionFromV1(bandResponse.data, nextTimeframe, bandSlot?.horizon ?? 5, priceRows) : null;
      const bandReady = Boolean(bandLatest && isActualBandPrediction(bandLatest) && checkBandOverlay(bandLatest).ok);
      const lineReady = lineSlot?.status === "active" && (lineHistory.length > 0 || Boolean(lineLatest));

      setLinePredictionHistory(lineHistory);
      setBandPredictionHistory(bandHistory);
      setPrediction(lineLatest);
      setBandPrediction(bandReady ? bandLatest : null);
      setBandEvaluation(null);
      setPredictionProvenance({
        latestRunId: lineSlot?.runId ?? null,
        selectedRunId: lineLatest?.run_id ?? lineHistory.at(-1)?.run_id ?? bandLatest?.run_id ?? null,
        isFallback: false,
      });

      if (priceRows.length === 0 && (lineReady || bandReady || bandHistory.length > 0)) {
        setAiState({
          kind: "empty",
          message: "예측 결과는 저장되어 있지만 가격 데이터가 없어 차트 위에 표시할 수 없습니다. 가격 데이터가 연결되면 표시됩니다.",
        });
      } else if (nextTimeframe === "1W" && bandReady) {
        setAiState({ kind: "ready", message: "자동 갱신된 1W AI 밴드를 표시합니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다." });
      } else if (lineReady && (bandReady || bandHistory.length > 0)) {
        setAiState({ kind: "ready", message: "1D 보수적 기준선과 1D AI 밴드를 저장된 v1 serving 기준으로 표시합니다." });
      } else if (!lineReady && (bandReady || bandHistory.length > 0)) {
        setAiState({ kind: "ready", message: nextTimeframe === "1W" ? "자동 갱신된 1W AI 밴드를 표시합니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다." : "자동 갱신된 AI 밴드를 표시합니다. 보수적 기준선 데이터는 아직 없습니다." });
      } else if (lineReady && !bandReady) {
        setAiState({ kind: "ready", message: "1D 보수적 기준선을 저장된 v1 serving 기준으로 표시합니다. AI 밴드 데이터는 아직 없습니다." });
      } else {
        setAiState({
          kind: "empty",
          message:
            nextTimeframe === "1W"
              ? "1W AI 밴드 데이터가 아직 없습니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다."
              : "v1 제품 슬롯은 준비됐지만 표시 가능한 예측 데이터가 없습니다.",
        });
      }
    } catch (error) {
      setPriceData([]);
      setPriceState({ kind: "error", message: extractErrorMessage(error) });
      setIndicatorData([]);
        setPrediction(null);
        setLinePredictionHistory([]);
        setBandPredictionHistory([]);
        setBandPrediction(null);
      setPredictionProvenance({
        latestRunId: null,
        selectedRunId: null,
        isFallback: false,
      });
      setBandEvaluation(null);
      setAiState({ kind: "empty", message: "가격 데이터 또는 예측 상태를 확인할 수 없습니다." });
      setErrorMessage(extractErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadStock("AAPL", "1D");
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextTicker = tickerInput.trim().toUpperCase() || "AAPL";
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadStock(nextTicker, timeframe);
    });
  }

  function handleTickerSelect(nextTicker: string) {
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadStock(nextTicker, timeframe);
    });
  }

  function handleTimeframeChange(nextTimeframe: DisplayTimeframe) {
    setTimeframe(nextTimeframe);
    startTransition(() => {
      void loadStock(selectedTicker, nextTimeframe);
    });
  }

  function handleIndicatorChange(id: IndicatorId, checked: boolean) {
    setSelectedIndicators((previous) => {
      if (checked) {
        return previous.includes(id) ? previous : [...previous, id];
      }
      return previous.filter((item) => item !== id);
    });
  }

  const latestPrice = getLastPrice(priceData);
  const latestPriceDate = latestPrice?.date ?? null;
  const changePercent = getChangePercent(priceData);
  const backendConfigWarning = getBackendConfigWarning();
  const hasPriceData = priceData.length > 0;
  const aiDisabled = timeframe === "1M";
  const currentLineSlot = getProductLineSlot(timeframe);
  const currentBandSlot = getProductBandSlot(timeframe);
  const activeLineModelDisplayName =
    LINE_OVERLAY_HELD && timeframe === "1D"
      ? "보수적 기준선 보류"
      : currentLineSlot?.status === "deferred"
      ? "1W 보수적 기준선 미제공"
      : currentLineSlot?.displayName ?? "-";
  const activeBandModelDisplayName = currentBandSlot?.displayName ?? "-";
  const forecastUnitLabel = timeframe === "1W" ? "주" : "거래일";
  const rawActivePrediction = prediction && timeframe !== "1M" && checkLineOverlay(prediction).ok ? prediction : null;
  const lineOutOfPriceRange = Boolean(
    hasPriceData &&
      rawActivePrediction &&
      rawActivePrediction.meta?.value_unit !== "return" &&
      !isPredictionLineWithinPriceRange(rawActivePrediction, priceData)
  );
  const activePrediction = rawActivePrediction && !lineOutOfPriceRange ? rawActivePrediction : null;
  const activeBandPrediction =
    bandPrediction && timeframe !== "1M" && isActualBandPrediction(bandPrediction) && checkBandOverlay(bandPrediction).ok
      ? bandPrediction
      : null;
  const activeLineHistory = useMemo(
    () =>
      timeframe === "1M"
        ? []
        : linePredictionHistory
            .filter((item) => isValidDate(item.asof_date) && Number.isFinite(item.value)),
    [linePredictionHistory, timeframe]
  );
  const activeBandHistory = useMemo(
    () =>
      timeframe === "1M"
        ? []
        : bandPredictionHistory.filter(
            (item) => isValidDate(item.asof_date) && Number.isFinite(item.lower) && Number.isFinite(item.upper)
          ),
    [bandPredictionHistory, timeframe]
  );
  const chartPriceData = priceData;
  const lineFutureForecastDates = useMemo(
    () => getFutureForecastDates(activePrediction, latestPriceDate),
    [activePrediction, latestPriceDate]
  );
  const bandFutureForecastDates = useMemo(
    () => getFutureForecastDates(activeBandPrediction, latestPriceDate),
    [activeBandPrediction, latestPriceDate]
  );
  const chartTimelineDates = useMemo(
    () =>
      uniqueDates([
          ...chartPriceData.map((row) => row.date),
          ...(lineFutureForecastDates.length >= 2 ? lineFutureForecastDates : []),
          ...(bandFutureForecastDates.length >= 2 ? bandFutureForecastDates : []),
          ...activeLineHistory.map((row) => row.forecast_date ?? row.asof_date),
          ...activeBandHistory.map((row) => row.asof_date),
        ]),
    [activeBandHistory, activeLineHistory, bandFutureForecastDates, chartPriceData, lineFutureForecastDates]
  );
  const chartActualDateSet = useMemo(() => new Set(chartPriceData.map((row) => row.date)), [chartPriceData]);
  const indicatorRowsInChartRange = useMemo(
    () => indicatorData.filter((row) => chartActualDateSet.has(row.date)),
    [chartActualDateSet, indicatorData]
  );
  const conservativeValue = getLastFinite(getPredictionLineValues(activePrediction));
  const lineAsofDate = activePrediction?.asof_date ?? activeLineHistory.at(-1)?.asof_date ?? null;
  const bandAsofDate = activeBandPrediction?.asof_date ?? activeBandHistory.at(-1)?.asof_date ?? null;
  const lineFreshness = getSlotFreshness(currentLineSlot, lineAsofDate, latestPriceDate, priceData);
  const bandFreshness = getSlotFreshness(currentBandSlot, bandAsofDate, latestPriceDate, priceData);
  const lineFuturePointCount = lineFutureForecastDates.length;
  const bandFuturePointCount = bandFutureForecastDates.length;
  const lineFutureHidden = Boolean(activePrediction && lineFuturePointCount < 2);
  const bandFutureHidden = Boolean(activeBandPrediction && bandFuturePointCount < 2);
  const changeClass = changePercent == null ? "is-flat" : changePercent < 0 ? "is-down" : "is-up";
  const bandLayerDisabled = aiDisabled || !hasPriceData || (!activeBandPrediction && activeBandHistory.length === 0);
  const lineLayerDisabled = aiDisabled || !hasPriceData || (!activePrediction && activeLineHistory.length === 0);
  const volumeLayerDisabled = priceData.every((row) => !Number.isFinite(row.volume) || row.volume == null || row.volume <= 0);
  const aiBandWidthPoints = useMemo(
    () => buildAiBandWidthPoints(activeBandHistory, activeBandPrediction).filter((point) => chartActualDateSet.has(point.date)),
    [activeBandHistory, activeBandPrediction, chartActualDateSet]
  );
  const availableIndicatorDefinitions = useMemo(() => {
    const dbDefinitions = INDICATOR_DEFINITIONS.filter((definition) => hasIndicatorValues(indicatorRowsInChartRange, definition));
    const aiBandWidthDefinition = INDICATOR_DEFINITIONS.find((definition) => definition.id === "ai_band_width");
    if (timeframe !== "1M" && aiBandWidthDefinition && (aiBandWidthPoints.length > 0 || selectedIndicators.includes("ai_band_width"))) {
      return [...dbDefinitions, aiBandWidthDefinition];
    }
    return dbDefinitions;
  }, [aiBandWidthPoints, indicatorRowsInChartRange, selectedIndicators, timeframe]);
  const visibleIndicatorSeries = useMemo(
    () =>
      buildVisibleIndicatorSeries({
        selectedIndicators,
        indicatorData: indicatorRowsInChartRange,
        indicatorErrorMessage,
        availableDefinitions: availableIndicatorDefinitions,
        allowedDates: chartActualDateSet,
        aiBandWidthPoints,
      }),
    [selectedIndicators, indicatorRowsInChartRange, indicatorErrorMessage, availableIndicatorDefinitions, chartActualDateSet, aiBandWidthPoints]
  );
  const indicatorTimelineDates = visibleTimelineDates.length > 0 ? visibleTimelineDates : chartTimelineDates;
  const isLegacyArtifact = isLegacyPrediction(activePrediction);
  const lineStatusLabel = aiDisabled
    ? "-"
    : LINE_OVERLAY_HELD && timeframe === "1D"
    ? "보류"
    : currentLineSlot?.status === "deferred"
    ? "deferred"
    : !hasPriceData && (rawActivePrediction || activeLineHistory.length > 0)
    ? "가격 데이터 연결 대기"
    : lineOutOfPriceRange
    ? `${formatFreshnessStatus(lineFreshness)} · 차트 숨김`
    : activePrediction || activeLineHistory.length > 0
    ? formatFreshnessStatus(lineFreshness)
    : "데이터 없음";
  const bandStatusLabel = aiDisabled
    ? "-"
    : !hasPriceData && (activeBandPrediction || activeBandHistory.length > 0)
    ? "가격 데이터 연결 대기"
    : activeBandPrediction || activeBandHistory.length > 0
    ? formatFreshnessStatus(bandFreshness)
    : currentBandSlot?.status === "active"
    ? "데이터 없음"
    : "-";
  const provenanceLabel = isLegacyArtifact
    ? "이전 실험 결과"
    : predictionProvenance.isFallback
    ? "저장된 예측 결과"
    : activePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0
    ? "v1 제품 슬롯"
    : "-";
  const provenanceSummary =
    rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0
      ? activeBandPrediction
        ? `${activeLineModelDisplayName}${lineOutOfPriceRange ? " 차트 숨김" : ""} / ${activeBandModelDisplayName}`
        : `${activeLineModelDisplayName}${lineOutOfPriceRange ? " 차트 숨김" : ""}`
      : "표시 중인 예측 결과가 없습니다.";
  const forecastHorizonCount =
    (rawActivePrediction ? currentLineSlot?.horizon : null) ??
    (activeBandPrediction ? currentBandSlot?.horizon : null) ??
    Math.max(lineFuturePointCount, bandFuturePointCount);
  const priceLatestDateLabel = latestPriceDate ?? "-";
  const bandAsofLabel = bandAsofDate ?? "-";
  const lineAsofLabel = lineAsofDate ?? "-";
  const futureForecastStatus =
    aiDisabled || !hasPriceData
      ? "-"
      : bandFuturePointCount >= 2
      ? `${bandFuturePointCount}${forecastUnitLabel}`
      : lineFuturePointCount >= 2
      ? `${lineFuturePointCount}${forecastUnitLabel}`
      : activeBandPrediction || activePrediction
      ? "미래 구간 부족"
      : "-";
  const staleNotice =
    bandFreshness === "stale"
      ? "AI 밴드는 마지막 저장된 기준으로 표시됩니다."
      : bandFreshness === "delayed"
      ? "AI 밴드는 가격 최신일보다 조금 이전 기준으로 표시됩니다."
      : null;
  const futureHiddenNotice =
    bandFutureHidden || lineFutureHidden
      ? "가격 최신일 이후의 forecast가 2개 미만이라 최신 미래 예측선은 숨기고 저장된 이력만 표시합니다."
      : null;
  const layerStatusNotice = staleNotice ?? futureHiddenNotice;
  const displayNotice = lineOutOfPriceRange
    ? "보수적 기준선이 현재 가격 범위를 벗어나 차트에서는 숨겼습니다. AI 밴드는 유지합니다."
    : !hasPriceData && (rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0)
    ? "예측 결과는 저장되어 있지만 가격 데이터가 없어 차트 위에 표시할 수 없습니다. 가격 데이터가 연결되면 표시됩니다."
    : layerStatusNotice
    ? layerStatusNotice
    : aiState.message;
  const emptyIndicatorMessage =
    availableIndicatorDefinitions.length === 0
      ? "이 타임프레임에 표시할 보조지표 데이터가 없습니다."
      : "오른쪽 메뉴에서 보조지표를 선택하세요.";
  const handleVisibleDatesChange = useCallback((dates: string[]) => {
    setVisibleTimelineDates((previous) => {
      if (previous.length === dates.length && previous.every((date, index) => date === dates[index])) {
        return previous;
      }
      return dates;
    });
  }, []);

  useEffect(() => {
    setVisibleTimelineDates(chartTimelineDates);
  }, [chartTimelineDates]);

  return (
    <div className="view-stack stock-view">
      <section className="stock-topbar">
        <p className="stock-intro">
          Lens는 가격 차트 위에 보수적 기준선과 자동 갱신된 AI 밴드를 겹쳐 보며, 리스크를 먼저 확인하는 투자 보조 도구입니다.
        </p>
        <form className="stock-topbar__form" onSubmit={handleSubmit}>
          <div className="ticker-box">
            <label className="ticker-field">
              <span>티커</span>
              <input value={tickerInput} onChange={(event) => setTickerInput(event.target.value.toUpperCase())} />
            </label>
            <button type="submit" className="primary-button primary-button--compact" disabled={isLoading}>
              조회
            </button>
          </div>

          <div className="quote-strip" aria-label="현재 시세">
            <div className="quote-item quote-item--ticker">
              <span>종목</span>
              <strong>{selectedTicker}</strong>
            </div>
            <div className="quote-item quote-item--price">
              <span>현재가</span>
              <strong>{formatNumber(latestPrice?.close)}</strong>
            </div>
            <div className={`quote-item quote-item--change ${changeClass}`}>
              <span>등락률</span>
              <strong>{formatPercent(changePercent)}</strong>
            </div>
            <div className="quote-item quote-item--volume">
              <span>거래량</span>
              <strong>{formatVolume(latestPrice?.volume)}</strong>
            </div>
          </div>

          <div className="stock-controls">
            <div className="segmented" aria-label="타임프레임">
              {TIMEFRAME_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={timeframe === option.value ? "is-active" : ""}
                  onClick={() => handleTimeframeChange(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <div className="segmented" aria-label="차트 타입">
              <button type="button" className={chartType === "candles" ? "is-active" : ""} onClick={() => setChartType("candles")}>
                캔들
              </button>
              <button type="button" className={chartType === "line" ? "is-active" : ""} onClick={() => setChartType("line")}>
                라인
              </button>
            </div>
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

      {backendConfigWarning ? <div className="notice">{backendConfigWarning}</div> : null}
      {errorMessage && priceData.length === 0 ? <div className="notice notice--error">{errorMessage}</div> : null}

      <section className="market-layout">
        <div className="chart-stack">
          <div className="chart-panel">
            {priceData.length > 0 ? (
              <Chart
                data={chartPriceData}
                ticker={selectedTicker}
                timeframe={timeframe}
                chartType={chartType}
                prediction={activePrediction}
                bandPrediction={activeBandPrediction}
                predictionHistory={activeLineHistory}
                bandPredictionHistory={activeBandHistory}
                layers={layers}
                timelineDates={chartTimelineDates}
                onVisibleDatesChange={handleVisibleDatesChange}
              />
            ) : (
              <div className="empty-state empty-state--stacked">
                <strong>가격 데이터 없음</strong>
                <span>{priceState.message}</span>
                {rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0 ? (
                  <span>저장된 AI 예측은 확인됐지만, 가격 데이터가 연결되면 차트 위에 표시됩니다.</span>
                ) : null}
              </div>
            )}
          </div>

          <IndicatorPanel
            series={visibleIndicatorSeries}
            note={indicatorErrorMessage}
            emptyMessage={emptyIndicatorMessage}
            timelineDates={indicatorTimelineDates}
          />
        </div>

        <aside className="panel side-panel forecast-panel">
          <div className="panel-heading">
            <div className="eyebrow">예측 레이어</div>
            <h2>모델 정보</h2>
          </div>

          <div className="provenance-card">
            <span className="eyebrow">예측 출처</span>
            <strong>{provenanceSummary}</strong>
            <div className="provenance-grid">
              <span>보수적 기준선</span>
              <strong>{activeLineModelDisplayName}</strong>
              <span>AI 밴드</span>
              <strong>{activeBandModelDisplayName}</strong>
              <span>가격 최신일</span>
              <strong>{priceLatestDateLabel}</strong>
              <span>AI 밴드 기준일</span>
              <strong>{bandAsofLabel}</strong>
              <span>AI 밴드 상태</span>
              <strong>
                <span className={`freshness-badge freshness-badge--${getFreshnessClass(bandFreshness)}`}>
                  {formatFreshnessStatus(bandFreshness)}
                </span>
              </strong>
              <span>보수적 기준선 기준일</span>
              <strong>{lineAsofLabel}</strong>
              <span>보수적 기준선 상태</span>
              <strong>
                <span className={`freshness-badge freshness-badge--${getFreshnessClass(lineFreshness)}`}>
                  {formatFreshnessStatus(lineFreshness)}
                </span>
              </strong>
              <span>예측 기간</span>
              <strong>{forecastHorizonCount ? `${forecastHorizonCount}${forecastUnitLabel}` : "-"}</strong>
              <span>미래 예측선</span>
              <strong>{futureForecastStatus}</strong>
            </div>
            {timeframe === "1D" && (rawActivePrediction || activeLineHistory.length > 0) ? (
              <p>1D 보수적 기준선은 저장된 v1 serving 기준으로 표시됩니다. raw output은 가격이 아니라 safe_line_score이며, 차트에서는 가격으로 환산합니다.</p>
            ) : null}
            {timeframe === "1W" ? <p>1W 보수적 기준선은 v1에서 제공하지 않습니다. 1W AI 밴드는 자동 갱신 결과를 표시합니다.</p> : null}
            {!aiDisabled && (rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0) ? (
              <details className="provenance-details">
                <summary>상세 정보</summary>
                <div className="provenance-grid provenance-grid--composite">
                  <span>표시 기준</span>
                  <strong>{provenanceLabel}</strong>
                  <span>보수적 기준선 상태</span>
                  <strong>{lineStatusLabel}</strong>
                  <span>밴드 상태</span>
                  <strong>{bandStatusLabel}</strong>
                  <span>보수적 기준선 ID</span>
                  <strong>{rawActivePrediction?.run_id ?? currentLineSlot?.runId ?? "준비 중"}</strong>
                  <span>밴드 실행 ID</span>
                  <strong>{activeBandPrediction?.run_id ?? activeBandHistory.at(-1)?.run_id ?? currentBandSlot?.runId ?? "-"}</strong>
                </div>
              </details>
            ) : null}
          </div>

          <div className="layer-group">
            <div className="layer-group__title">가격 오버레이</div>
            <LayerToggle
              label="보수적 기준선"
              checked={!lineLayerDisabled && layers.conservativeLine}
              disabled={lineLayerDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, conservativeLine: checked }))}
            />
            <p className="layer-description">
              {timeframe === "1W"
                ? "1W 보수적 기준선은 v1에서 제공하지 않습니다."
                : "1D 보수적 기준선은 저장된 v1 serving 기준입니다. raw output은 가격이 아니라 safe_line_score이며, asof 종가 × (1 + safe_line_score)로 환산합니다."}
            </p>
            <LayerToggle
              label="AI 밴드"
              checked={!bandLayerDisabled && layers.aiBand}
              disabled={bandLayerDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, aiBand: checked }))}
            />
            <p className="layer-description">
              {bandFreshness === "stale"
                ? "AI 밴드는 마지막 저장된 기준으로 표시됩니다."
                : timeframe === "1W"
                ? "자동 갱신된 1W AI 밴드입니다. asof 가격에 밴드 수익률을 곱해 가격 차트 위에 표시합니다."
                : "자동 갱신된 1D AI 밴드입니다. 최신 parquet 결과를 가격 차트 위에 표시합니다."}
            </p>
            {!bandLayerDisabled && layers.aiBand ? (
              <div className="layer-metrics">
                <div>
                  <span>커버리지</span>
                  <strong>{formatRatio(bandEvaluation?.coverage)}</strong>
                </div>
                <div>
                  <span>평균 밴드 폭</span>
                  <strong>{formatNumber(bandEvaluation?.avg_band_width)}</strong>
                </div>
              </div>
            ) : null}
            {!lineLayerDisabled && layers.conservativeLine ? (
              <div className="layer-metrics">
                <div>
                  <span>예측가</span>
                  <strong>{formatNumber(conservativeValue)}</strong>
                </div>
                <div>
                  <span>상태</span>
                  <strong>{formatFreshnessStatus(lineFreshness)}</strong>
                </div>
              </div>
            ) : null}
          </div>

          <div className="layer-group">
            <div className="layer-group__title">거래량</div>
            <LayerToggle
              label="거래량 bar"
              checked={!volumeLayerDisabled && layers.volumeBar}
              disabled={volumeLayerDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, volumeBar: checked }))}
            />
          </div>

          <div className="layer-group">
            <details className="indicator-menu" open>
              <summary>
                <span>
                  <span className="eyebrow">하단 지표</span>
                  <strong>하단 지표 선택</strong>
                </span>
              </summary>
              <div className="indicator-selector">
                {["기본", "추세", "변동성/위치", "AI"].map((category) => {
                  const options = availableIndicatorDefinitions.filter((definition) => definition.category === category);
                  if (options.length === 0) {
                    return null;
                  }
                  return (
                    <IndicatorOptionGroup
                      key={category}
                      title={category}
                      options={options}
                      selectedIndicators={selectedIndicators}
                      onChange={handleIndicatorChange}
                    />
                  );
                })}
                {availableIndicatorDefinitions.length === 0 ? (
                  <div className="compact-note">표시할 수 있는 보조지표 값이 없습니다.</div>
                ) : null}
              </div>
            </details>
          </div>

          {lineOutOfPriceRange || aiState.kind !== "ready" ? <div className="notice">{displayNotice}</div> : null}
        </aside>
      </section>
    </div>
  );
}

interface IndicatorOptionGroupProps {
  title: string;
  options: IndicatorDefinition[];
  selectedIndicators: IndicatorId[];
  onChange: (id: IndicatorId, checked: boolean) => void;
}

function IndicatorOptionGroup({ title, options, selectedIndicators, onChange }: IndicatorOptionGroupProps) {
  return (
    <div className="indicator-selector__group">
      <span>{title}</span>
      {options.map((option) => (
        <IndicatorOption
          key={option.id}
          option={option}
          checked={selectedIndicators.includes(option.id)}
          onChange={onChange}
        />
      ))}
    </div>
  );
}

interface IndicatorOptionProps {
  option: IndicatorDefinition;
  checked: boolean;
  onChange: (id: IndicatorId, checked: boolean) => void;
}

function IndicatorOption({ option, checked, onChange }: IndicatorOptionProps) {
  return (
    <label className={`indicator-option${checked ? " is-on" : ""}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(option.id, event.target.checked)}
      />
      <span className="indicator-option__dot" aria-hidden="true" />
      <span className="indicator-option__body">
        <span className="indicator-option__label">{option.label}</span>
        <span className="indicator-option__description">{option.description}</span>
      </span>
    </label>
  );
}
