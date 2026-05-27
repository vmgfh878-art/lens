import type { IChartApi } from "lightweight-charts";

import type {
  PredictionResult,
  PriceBar,
  ProductBandHistoryPoint,
  ProductLineHistoryPoint,
} from "@/api/client";

import { MAX_ROLLING_HISTORY_GAP_DAYS, OverlayPoint, OverlayState, VolumePoint } from "./types";

// ── 시간 / 비교 ─────────────────────────────────────────────

export function isValidTime(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && !Number.isNaN(Date.parse(value));
}

export function sortUniqueByTime<T extends { time: string }>(rows: T[]) {
  const deduped = new Map<string, T>();
  rows.forEach((row) => {
    if (isValidTime(row.time)) {
      deduped.set(row.time, row);
    }
  });
  return Array.from(deduped.values()).sort((left, right) => left.time.localeCompare(right.time));
}

export function sanitizeOverlayPoints(points: OverlayPoint[]) {
  return sortUniqueByTime(points.filter((point) => Number.isFinite(point.value)));
}

function daysBetween(left: string, right: string) {
  const leftTime = Date.parse(left);
  const rightTime = Date.parse(right);
  if (!Number.isFinite(leftTime) || !Number.isFinite(rightTime)) {
    return 0;
  }
  return Math.abs(rightTime - leftTime) / (24 * 60 * 60 * 1000);
}

export function keepLatestContiguousHistory(points: OverlayPoint[]) {
  const sorted = sanitizeOverlayPoints(points);
  if (sorted.length <= 1) {
    return sorted;
  }

  const segments: OverlayPoint[][] = [];
  let currentSegment: OverlayPoint[] = [sorted[0]];
  for (let index = 1; index < sorted.length; index += 1) {
    if (daysBetween(sorted[index - 1].time, sorted[index].time) > MAX_ROLLING_HISTORY_GAP_DAYS) {
      segments.push(currentSegment);
      currentSegment = [sorted[index]];
    } else {
      currentSegment.push(sorted[index]);
    }
  }
  segments.push(currentSegment);

  for (let index = segments.length - 1; index >= 0; index -= 1) {
    if (segments[index].length >= 2) {
      return segments[index];
    }
  }
  return segments[segments.length - 1] ?? [];
}

export function sanitizeWhitespaceDates(rows: Array<{ time: string }>, occupiedTimes: Set<string>) {
  return sortUniqueByTime(rows).filter((row) => !occupiedTimes.has(row.time));
}

export function compareDate(left: string | null | undefined, right: string | null | undefined) {
  if (!left && !right) return 0;
  if (!left) return -1;
  if (!right) return 1;
  return left.localeCompare(right);
}

export function maxDate(dates: Array<string | null | undefined>) {
  const validDates = dates.filter(isValidTime).sort((left, right) => left.localeCompare(right));
  return validDates.length > 0 ? validDates[validDates.length - 1] : null;
}

export function getLatestPriceDate(rows: PriceBar[]) {
  return maxDate(rows.map((row) => row.date));
}

// ── prediction meta 해석 ──────────────────────────────────

function getPredictionMetaString(prediction: PredictionResult | null | undefined, key: string) {
  const value = prediction?.meta?.[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

export function isActualBandPrediction(prediction: PredictionResult | null | undefined) {
  if (!prediction) return false;

  const role = getPredictionMetaString(prediction, "role");
  const bandFieldsPolicy = getPredictionMetaString(prediction, "band_fields_policy");
  const bandSavedInCp140 = prediction.meta?.band_saved_in_cp140;

  if (role !== "band_model") return false;
  if (bandSavedInCp140 === false || bandSavedInCp140 === "false") return false;
  return bandFieldsPolicy !== "schema_required_degenerate_equal_to_line";
}

// ── 시리즈 빌더 (candle/line/volume) ────────────────────

export function buildCandleData(rows: PriceBar[], whitespaceDates: Array<{ time: string }>) {
  const candles = sortUniqueByTime(
    rows
      .map((item) => {
        const close = item.close;
        const open = item.open ?? close;
        const high = item.high ?? close;
        const low = item.low ?? close;
        if (
          !isValidTime(item.date) ||
          !Number.isFinite(open) ||
          !Number.isFinite(high) ||
          !Number.isFinite(low) ||
          !Number.isFinite(close)
        ) {
          return null;
        }
        return { time: item.date, open, high, low, close };
      })
      .filter(
        (item): item is { time: string; open: number; high: number; low: number; close: number } =>
          item !== null
      )
  );
  const occupiedTimes = new Set(candles.map((item) => item.time));
  return [...candles, ...sanitizeWhitespaceDates(whitespaceDates, occupiedTimes)].sort((left, right) =>
    left.time.localeCompare(right.time)
  );
}

export function buildPriceLineData(rows: PriceBar[], whitespaceDates: Array<{ time: string }>) {
  const lineRows = sortUniqueByTime(
    rows
      .map((item) => {
        if (!isValidTime(item.date) || !Number.isFinite(item.close)) return null;
        return { time: item.date, value: item.close };
      })
      .filter((item): item is OverlayPoint => item !== null)
  );
  const occupiedTimes = new Set(lineRows.map((item) => item.time));
  return [...lineRows, ...sanitizeWhitespaceDates(whitespaceDates, occupiedTimes)].sort((left, right) =>
    left.time.localeCompare(right.time)
  );
}

export function buildVolumeData(rows: PriceBar[]): VolumePoint[] {
  return sortUniqueByTime(
    rows
      .map((item) => {
        if (
          !isValidTime(item.date) ||
          !Number.isFinite(item.volume) ||
          item.volume == null ||
          item.volume <= 0
        ) {
          return null;
        }
        const close = item.close;
        const open = item.open ?? close;
        const color = close >= open ? "rgba(15, 159, 110, 0.24)" : "rgba(217, 45, 32, 0.22)";
        return { time: item.date, value: item.volume, color };
      })
      .filter((item): item is VolumePoint => item !== null)
  );
}

// ── 표시 범위 / 가시 logical range ────────────────────────

export function getInitialVisibleCount(timeframe: "1D" | "1W" | "1M") {
  if (timeframe === "1D") return 260;
  if (timeframe === "1W") return 156;
  return 84;
}

export function applyInitialVisibleRange(
  chart: IChartApi,
  timelineDates: string[],
  timeframe: "1D" | "1W" | "1M"
) {
  if (timelineDates.length === 0) return;
  const visibleCount = Math.min(getInitialVisibleCount(timeframe), timelineDates.length);
  const to = timelineDates.length - 1;
  const from = Math.max(0, to - visibleCount + 1);
  chart.timeScale().setVisibleLogicalRange({ from, to });
}

export function emitVisibleDates(
  chart: IChartApi,
  timelineDates: string[],
  onVisibleDatesChange?: (dates: string[]) => void
) {
  if (!onVisibleDatesChange || timelineDates.length === 0) return;
  const range = chart.timeScale().getVisibleLogicalRange();
  if (!range) {
    onVisibleDatesChange(timelineDates);
    return;
  }
  const from = Math.max(0, Math.floor(range.from));
  const to = Math.min(timelineDates.length - 1, Math.ceil(range.to));
  onVisibleDatesChange(timelineDates.slice(from, to + 1));
}

export function shouldUseSeparatePredictionScale(data: PriceBar[], overlayPoints: OverlayPoint[]) {
  const prices = data.flatMap((item) => [
    item.open ?? item.close,
    item.high ?? item.close,
    item.low ?? item.close,
    item.close,
  ]);
  const finitePrices = prices.filter(Number.isFinite);
  const overlayValues = overlayPoints.map((point) => point.value).filter(Number.isFinite);
  if (finitePrices.length < 2 || overlayValues.length < 2) return false;

  const priceMin = Math.min(...finitePrices);
  const priceMax = Math.max(...finitePrices);
  const overlayMin = Math.min(...overlayValues);
  const overlayMax = Math.max(...overlayValues);
  const priceSpan = Math.max(priceMax - priceMin, Math.abs(priceMax) * 0.05, 1);
  const overlaySpan = overlayMax - overlayMin;
  return (
    overlaySpan > priceSpan * 4 ||
    overlayMin < priceMin - priceSpan * 2 ||
    overlayMax > priceMax + priceSpan * 2
  );
}

// ── overlay state 빌드 (line / band 예측 시각화 데이터) ─

function hasSameLength(dates: string[], values: number[] | null | undefined) {
  return Boolean(values && dates.length > 0 && dates.length === values.length);
}

function getConservativeLineValues(prediction: PredictionResult | null | undefined) {
  if (!prediction) return [];
  return prediction.conservative_series?.length
    ? prediction.conservative_series
    : prediction.line_series ?? [];
}

function buildFutureOverlayData(params: {
  dates: string[];
  values: number[];
  asofDate?: string | null;
  latestPriceDate: string | null;
}) {
  const { dates, values, asofDate, latestPriceDate } = params;
  if (dates.length === 0 || dates.length !== values.length) return [];

  const cutoffDate = maxDate([latestPriceDate, asofDate]);
  return sanitizeOverlayPoints(
    dates
      .map((date, index) => {
        if (!isValidTime(date) || !Number.isFinite(values[index])) return null;
        if (cutoffDate && compareDate(date, cutoffDate) <= 0) return null;
        return { time: date, value: values[index] };
      })
      .filter((point): point is OverlayPoint => point !== null)
  );
}

function buildRollingConservativeHistory(
  history: ProductLineHistoryPoint[] | undefined,
  latestPriceDate: string | null
): OverlayPoint[] {
  return keepLatestContiguousHistory(
    (history ?? [])
      .map((item) => {
        const displayDate = item.forecast_date ?? item.asof_date;
        if (!isValidTime(displayDate) || !Number.isFinite(item.value)) return null;
        if (latestPriceDate && compareDate(displayDate, latestPriceDate) > 0) return null;
        return { time: displayDate, value: item.value };
      })
      .filter((point): point is OverlayPoint => point !== null)
  );
}

function buildRollingBandHistory(
  history: ProductBandHistoryPoint[] | undefined,
  field: "upper" | "lower",
  latestPriceDate: string | null
): OverlayPoint[] {
  return keepLatestContiguousHistory(
    (history ?? [])
      .map((item) => {
        const value = item[field];
        if (!isValidTime(item.asof_date) || !Number.isFinite(value)) return null;
        if (latestPriceDate && compareDate(item.asof_date, latestPriceDate) > 0) return null;
        return { time: item.asof_date, value };
      })
      .filter((point): point is OverlayPoint => point !== null)
  );
}

function getLatestBandHistoryDate(history: ProductBandHistoryPoint[] | undefined) {
  return maxDate((history ?? []).map((item) => item.asof_date));
}

function emptyOverlayState(warning: string | null = null): OverlayState {
  return {
    canDrawBand: false,
    canDrawConservativeLine: false,
    conservativeData: [],
    conservativeHistoryData: [],
    upperBandData: [],
    lowerBandData: [],
    upperBandHistoryData: [],
    lowerBandHistoryData: [],
    modelMarkerDate: null,
    warning,
  };
}

export function buildOverlayState(
  prediction: PredictionResult | null | undefined,
  bandPrediction: PredictionResult | null | undefined,
  predictionHistory: ProductLineHistoryPoint[] | undefined,
  bandPredictionHistory: ProductBandHistoryPoint[] | undefined,
  layers: { aiBand: boolean; conservativeLine: boolean; volumeBar: boolean } | undefined,
  timeframe: "1D" | "1W" | "1M",
  latestPriceDate: string | null
): OverlayState {
  const hasHistory = Boolean(predictionHistory?.length || bandPredictionHistory?.length);
  if (timeframe === "1M" || (!prediction && !bandPrediction && !hasHistory)) {
    return emptyOverlayState();
  }

  const actualBandPrediction = isActualBandPrediction(bandPrediction) ? bandPrediction : null;
  const lineForecastDates = prediction?.forecast_dates ?? [];
  const bandForecastDates = actualBandPrediction?.forecast_dates ?? [];
  const conservativeValues = getConservativeLineValues(prediction);
  const hasConservativeLine = hasSameLength(lineForecastDates, conservativeValues);
  const hasUpper = hasSameLength(bandForecastDates, actualBandPrediction?.upper_band_series);
  const hasLower = hasSameLength(bandForecastDates, actualBandPrediction?.lower_band_series);
  const modelMarkerDate =
    actualBandPrediction?.asof_date ?? getLatestBandHistoryDate(bandPredictionHistory);

  if ((prediction && !hasConservativeLine) || (actualBandPrediction && (!hasUpper || !hasLower))) {
    return {
      ...emptyOverlayState("예측 날짜와 시리즈 길이가 맞지 않아 예측 레이어를 숨겼습니다."),
      modelMarkerDate,
    };
  }

  const conservativeData = hasConservativeLine
    ? buildFutureOverlayData({
        dates: lineForecastDates,
        values: conservativeValues,
        asofDate: prediction?.asof_date,
        latestPriceDate,
      })
    : [];
  const upperBandData =
    hasUpper && actualBandPrediction
      ? buildFutureOverlayData({
          dates: bandForecastDates,
          values: actualBandPrediction.upper_band_series,
          asofDate: actualBandPrediction.asof_date,
          latestPriceDate,
        })
      : [];
  const lowerBandData =
    hasLower && actualBandPrediction
      ? buildFutureOverlayData({
          dates: bandForecastDates,
          values: actualBandPrediction.lower_band_series,
          asofDate: actualBandPrediction.asof_date,
          latestPriceDate,
        })
      : [];

  const conservativeHistoryData = buildRollingConservativeHistory(predictionHistory, latestPriceDate);
  const upperBandHistoryData = buildRollingBandHistory(bandPredictionHistory, "upper", latestPriceDate);
  const lowerBandHistoryData = buildRollingBandHistory(bandPredictionHistory, "lower", latestPriceDate);

  const canDrawConservativeLine =
    Boolean(layers?.conservativeLine) &&
    (conservativeData.length >= 2 || conservativeHistoryData.length >= 2);
  const canDrawBand =
    Boolean(layers?.aiBand) &&
    ((upperBandData.length >= 2 && lowerBandData.length >= 2) ||
      (upperBandHistoryData.length >= 2 && lowerBandHistoryData.length >= 2));
  const warning =
    (prediction && conservativeData.length === 0 && conservativeHistoryData.length === 0) ||
    (actualBandPrediction &&
      upperBandData.length === 0 &&
      lowerBandData.length === 0 &&
      upperBandHistoryData.length === 0)
      ? "최신 가격 이후에 표시할 예측 날짜가 없어 예측 레이어를 숨겼습니다."
      : null;

  return {
    canDrawBand,
    canDrawConservativeLine,
    conservativeData,
    conservativeHistoryData,
    upperBandData,
    lowerBandData,
    upperBandHistoryData,
    lowerBandHistoryData,
    modelMarkerDate,
    warning,
  };
}
