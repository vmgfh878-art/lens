"use client";

import { useEffect, useMemo, useRef } from "react";
import { createChart, IChartApi, LineStyle } from "lightweight-charts";

import { PredictionResult, PriceBar, ProductBandHistoryPoint, ProductLineHistoryPoint } from "@/api/client";

interface ChartProps {
  data: PriceBar[];
  ticker: string;
  timeframe: "1D" | "1W" | "1M";
  chartType: "candles" | "line";
  prediction?: PredictionResult | null;
  bandPrediction?: PredictionResult | null;
  predictionHistory?: ProductLineHistoryPoint[];
  bandPredictionHistory?: ProductBandHistoryPoint[];
  layers?: {
    aiBand: boolean;
    conservativeLine: boolean;
    volumeBar: boolean;
  };
  timelineDates?: string[];
  onVisibleDatesChange?: (dates: string[]) => void;
}

interface OverlayPoint {
  time: string;
  value: number;
}

interface VolumePoint extends OverlayPoint {
  color: string;
}

interface OverlayState {
  canDrawBand: boolean;
  canDrawConservativeLine: boolean;
  conservativeData: OverlayPoint[];
  conservativeHistoryData: OverlayPoint[];
  upperBandData: OverlayPoint[];
  lowerBandData: OverlayPoint[];
  upperBandHistoryData: OverlayPoint[];
  lowerBandHistoryData: OverlayPoint[];
  modelMarkerDate: string | null;
  warning: string | null;
}

const PREDICTION_SCALE_ID = "prediction-overlay";
const MAX_ROLLING_HISTORY_GAP_DAYS = 10;

function isValidTime(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && !Number.isNaN(Date.parse(value));
}

function sortUniqueByTime<T extends { time: string }>(rows: T[]) {
  const deduped = new Map<string, T>();
  rows.forEach((row) => {
    if (isValidTime(row.time)) {
      deduped.set(row.time, row);
    }
  });
  return Array.from(deduped.values()).sort((left, right) => left.time.localeCompare(right.time));
}

function sanitizeOverlayPoints(points: OverlayPoint[]) {
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

function keepLatestContiguousHistory(points: OverlayPoint[]) {
  const sorted = sanitizeOverlayPoints(points);
  let segmentStart = 0;
  for (let index = 1; index < sorted.length; index += 1) {
    if (daysBetween(sorted[index - 1].time, sorted[index].time) > MAX_ROLLING_HISTORY_GAP_DAYS) {
      segmentStart = index;
    }
  }
  return sorted.slice(segmentStart);
}

function sanitizeWhitespaceDates(rows: Array<{ time: string }>, occupiedTimes: Set<string>) {
  return sortUniqueByTime(rows).filter((row) => !occupiedTimes.has(row.time));
}

function compareDate(left: string | null | undefined, right: string | null | undefined) {
  if (!left && !right) {
    return 0;
  }
  if (!left) {
    return -1;
  }
  if (!right) {
    return 1;
  }
  return left.localeCompare(right);
}

function maxDate(dates: Array<string | null | undefined>) {
  const validDates = dates.filter(isValidTime).sort((left, right) => left.localeCompare(right));
  return validDates.length > 0 ? validDates[validDates.length - 1] : null;
}

function getLatestPriceDate(rows: PriceBar[]) {
  return maxDate(rows.map((row) => row.date));
}

function getPredictionMetaString(prediction: PredictionResult | null | undefined, key: string) {
  const value = prediction?.meta?.[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function isActualBandPrediction(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return false;
  }

  const role = getPredictionMetaString(prediction, "role");
  const bandFieldsPolicy = getPredictionMetaString(prediction, "band_fields_policy");
  const bandSavedInCp140 = prediction.meta?.band_saved_in_cp140;

  if (role !== "band_model") {
    return false;
  }

  if (bandSavedInCp140 === false || bandSavedInCp140 === "false") {
    return false;
  }

  return bandFieldsPolicy !== "schema_required_degenerate_equal_to_line";
}

function buildCandleData(rows: PriceBar[], whitespaceDates: Array<{ time: string }>) {
  const candles = sortUniqueByTime(
    rows
      .map((item) => {
        const close = item.close;
        const open = item.open ?? close;
        const high = item.high ?? close;
        const low = item.low ?? close;
        if (!isValidTime(item.date) || !Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) {
          return null;
        }
        return {
          time: item.date,
          open,
          high,
          low,
          close,
        };
      })
      .filter((item): item is { time: string; open: number; high: number; low: number; close: number } => item !== null)
  );
  const occupiedTimes = new Set(candles.map((item) => item.time));
  return [...candles, ...sanitizeWhitespaceDates(whitespaceDates, occupiedTimes)].sort((left, right) =>
    left.time.localeCompare(right.time)
  );
}

function buildPriceLineData(rows: PriceBar[], whitespaceDates: Array<{ time: string }>) {
  const lineRows = sortUniqueByTime(
    rows
      .map((item) => {
        if (!isValidTime(item.date) || !Number.isFinite(item.close)) {
          return null;
        }
        return {
          time: item.date,
          value: item.close,
        };
      })
      .filter((item): item is OverlayPoint => item !== null)
  );
  const occupiedTimes = new Set(lineRows.map((item) => item.time));
  return [...lineRows, ...sanitizeWhitespaceDates(whitespaceDates, occupiedTimes)].sort((left, right) =>
    left.time.localeCompare(right.time)
  );
}

function buildVolumeData(rows: PriceBar[]): VolumePoint[] {
  return sortUniqueByTime(
    rows
      .map((item) => {
        if (!isValidTime(item.date) || !Number.isFinite(item.volume) || item.volume == null || item.volume <= 0) {
          return null;
        }
        const close = item.close;
        const open = item.open ?? close;
        const color = close >= open ? "rgba(15, 159, 110, 0.24)" : "rgba(217, 45, 32, 0.22)";
        return {
          time: item.date,
          value: item.volume,
          color,
        };
      })
      .filter((item): item is VolumePoint => item !== null)
  );
}

function getInitialVisibleCount(timeframe: ChartProps["timeframe"]) {
  if (timeframe === "1D") {
    return 260;
  }
  if (timeframe === "1W") {
    return 156;
  }
  return 84;
}

function applyInitialVisibleRange(chart: IChartApi, timelineDates: string[], timeframe: ChartProps["timeframe"]) {
  if (timelineDates.length === 0) {
    return;
  }
  const visibleCount = Math.min(getInitialVisibleCount(timeframe), timelineDates.length);
  const to = timelineDates.length - 1;
  const from = Math.max(0, to - visibleCount + 1);
  chart.timeScale().setVisibleLogicalRange({ from, to });
}

function hasSameLength(dates: string[], values: number[] | null | undefined) {
  return Boolean(values && dates.length > 0 && dates.length === values.length);
}

function getConservativeLineValues(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return [];
  }
  return prediction.conservative_series?.length ? prediction.conservative_series : prediction.line_series ?? [];
}

function buildFutureOverlayData(params: {
  dates: string[];
  values: number[];
  asofDate?: string | null;
  latestPriceDate: string | null;
}) {
  const { dates, values, asofDate, latestPriceDate } = params;
  if (dates.length === 0 || dates.length !== values.length) {
    return [];
  }

  const cutoffDate = maxDate([latestPriceDate, asofDate]);
  return sanitizeOverlayPoints(
    dates
      .map((date, index) => {
        if (!isValidTime(date) || !Number.isFinite(values[index])) {
          return null;
        }
        if (cutoffDate && compareDate(date, cutoffDate) <= 0) {
          return null;
        }
        return {
          time: date,
          value: values[index],
        };
      })
      .filter((point): point is OverlayPoint => point !== null)
  );
}

function buildRollingConservativeHistory(history: ProductLineHistoryPoint[] | undefined, latestPriceDate: string | null): OverlayPoint[] {
  return keepLatestContiguousHistory(
    (history ?? [])
      .map((item) => {
        if (!isValidTime(item.asof_date) || !Number.isFinite(item.value)) {
          return null;
        }
        if (latestPriceDate && compareDate(item.asof_date, latestPriceDate) > 0) {
          return null;
        }
        return { time: item.asof_date, value: item.value };
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
        if (!isValidTime(item.asof_date) || !Number.isFinite(value)) {
          return null;
        }
        if (latestPriceDate && compareDate(item.asof_date, latestPriceDate) > 0) {
          return null;
        }
        return { time: item.asof_date, value };
      })
      .filter((point): point is OverlayPoint => point !== null)
  );
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

function buildOverlayState(
  prediction: PredictionResult | null | undefined,
  bandPrediction: PredictionResult | null | undefined,
  predictionHistory: ProductLineHistoryPoint[] | undefined,
  bandPredictionHistory: ProductBandHistoryPoint[] | undefined,
  layers: ChartProps["layers"],
  timeframe: ChartProps["timeframe"],
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
  const modelMarkerDate = prediction?.asof_date ?? actualBandPrediction?.asof_date ?? null;

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
    Boolean(layers?.conservativeLine) && (conservativeData.length >= 2 || conservativeHistoryData.length >= 2);
  const canDrawBand =
    Boolean(layers?.aiBand) &&
    ((upperBandData.length >= 2 && lowerBandData.length >= 2) ||
      (upperBandHistoryData.length >= 2 && lowerBandHistoryData.length >= 2));
  const warning =
    (prediction && conservativeData.length === 0 && conservativeHistoryData.length === 0) ||
    (actualBandPrediction && upperBandData.length === 0 && lowerBandData.length === 0 && upperBandHistoryData.length === 0)
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

function emitVisibleDates(
  chart: IChartApi,
  timelineDates: string[],
  onVisibleDatesChange?: (dates: string[]) => void
) {
  if (!onVisibleDatesChange || timelineDates.length === 0) {
    return;
  }

  const range = chart.timeScale().getVisibleLogicalRange();
  if (!range) {
    onVisibleDatesChange(timelineDates);
    return;
  }

  const from = Math.max(0, Math.floor(range.from));
  const to = Math.min(timelineDates.length - 1, Math.ceil(range.to));
  onVisibleDatesChange(timelineDates.slice(from, to + 1));
}

function shouldUseSeparatePredictionScale(data: PriceBar[], overlayPoints: OverlayPoint[]) {
  const prices = data.flatMap((item) => [item.open ?? item.close, item.high ?? item.close, item.low ?? item.close, item.close]);
  const finitePrices = prices.filter(Number.isFinite);
  const overlayValues = overlayPoints.map((point) => point.value).filter(Number.isFinite);
  if (finitePrices.length < 2 || overlayValues.length < 2) {
    return false;
  }

  const priceMin = Math.min(...finitePrices);
  const priceMax = Math.max(...finitePrices);
  const overlayMin = Math.min(...overlayValues);
  const overlayMax = Math.max(...overlayValues);
  const priceSpan = Math.max(priceMax - priceMin, Math.abs(priceMax) * 0.05, 1);
  const overlaySpan = overlayMax - overlayMin;
  return overlaySpan > priceSpan * 4 || overlayMin < priceMin - priceSpan * 2 || overlayMax > priceMax + priceSpan * 2;
}

export default function Chart({
  data,
  ticker,
  timeframe,
  chartType,
  prediction,
  bandPrediction,
  predictionHistory,
  bandPredictionHistory,
  layers,
  timelineDates,
  onVisibleDatesChange,
}: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const forecastMarkerRef = useRef<HTMLDivElement>(null);
  const latestPriceDate = useMemo(() => getLatestPriceDate(data), [data]);
  const overlayState = useMemo(
    () => buildOverlayState(prediction, bandPrediction, predictionHistory, bandPredictionHistory, layers, timeframe, latestPriceDate),
    [bandPrediction, bandPredictionHistory, latestPriceDate, prediction, predictionHistory, layers, timeframe]
  );
  const {
    canDrawBand,
    canDrawConservativeLine,
    conservativeData,
    conservativeHistoryData,
    lowerBandData,
    lowerBandHistoryData,
    modelMarkerDate,
    upperBandData,
    upperBandHistoryData,
    warning,
  } = overlayState;
  const markerDate = timeframe !== "1M" && (canDrawBand || canDrawConservativeLine) ? modelMarkerDate : null;
  const latestForecastLabel =
    timeframe === "1W"
      ? `최신 ${Math.max(conservativeData.length, 1)}주 예측`
      : `최신 ${Math.max(conservativeData.length, 1)}일 예측`;
  const chartTimelineDates = useMemo(() => {
    const source = timelineDates && timelineDates.length > 0 ? timelineDates : data.map((item) => item.date);
    const dates = markerDate ? [...source, markerDate] : source;
    return sortUniqueByTime(dates.map((time) => ({ time }))).map((item) => item.time);
  }, [data, timelineDates, markerDate]);
  const priceDateSet = useMemo(() => new Set(sortUniqueByTime(data.map((item) => ({ time: item.date }))).map((item) => item.time)), [data]);
  const whitespaceDates = useMemo(
    () => chartTimelineDates.filter((date) => !priceDateSet.has(date)).map((time) => ({ time })),
    [chartTimelineDates, priceDateSet]
  );
  const volumeData = useMemo(() => buildVolumeData(data), [data]);
  const useSeparatePredictionScale = useMemo(
    () =>
      shouldUseSeparatePredictionScale(data, [
        ...conservativeData,
        ...conservativeHistoryData,
        ...upperBandData,
        ...upperBandHistoryData,
        ...lowerBandData,
        ...lowerBandHistoryData,
      ]),
    [
      conservativeData,
      conservativeHistoryData,
      data,
      lowerBandData,
      lowerBandHistoryData,
      upperBandData,
      upperBandHistoryData,
    ]
  );

  useEffect(() => {
    if (!containerRef.current || data.length === 0) {
      return;
    }

    const chartHeight = containerRef.current.clientHeight || 560;
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: chartHeight,
      layout: {
        background: { color: "#ffffff" },
        textColor: "#374151",
      },
      grid: {
        vertLines: { color: "rgba(17, 24, 39, 0.05)" },
        horzLines: { color: "rgba(17, 24, 39, 0.05)" },
      },
      rightPriceScale: {
        borderColor: "rgba(17, 24, 39, 0.12)",
      },
      timeScale: {
        borderColor: "rgba(17, 24, 39, 0.12)",
      },
      crosshair: {
        mode: 1,
      },
    });
    if (useSeparatePredictionScale) {
      chart.priceScale(PREDICTION_SCALE_ID).applyOptions({
        scaleMargins: {
          top: 0.08,
          bottom: 0.08,
        },
      });
    }

    const updateForecastMarker = () => {
      const marker = forecastMarkerRef.current;
      if (!marker) {
        return;
      }
      if (!markerDate) {
        marker.style.display = "none";
        return;
      }

      const coordinate = chart.timeScale().timeToCoordinate(markerDate);
      const width = containerRef.current?.clientWidth ?? 0;
      if (coordinate == null || coordinate < 0 || coordinate > width) {
        marker.style.display = "none";
        return;
      }
      marker.style.display = "block";
      marker.style.left = `${coordinate}px`;
    };

    const handleVisibleRangeChange = () => {
      emitVisibleDates(chart, chartTimelineDates, onVisibleDatesChange);
      updateForecastMarker();
    };

    if (chartType === "candles") {
      const candleSeries = chart.addCandlestickSeries({
        upColor: "#0f9f6e",
        downColor: "#d92d20",
        wickUpColor: "#0f9f6e",
        wickDownColor: "#d92d20",
        borderVisible: false,
      });

      candleSeries.setData(buildCandleData(data, whitespaceDates));
    } else {
      const lineSeries = chart.addLineSeries({
        color: "#1f2937",
        lineWidth: 2,
      });
      lineSeries.setData(buildPriceLineData(data, whitespaceDates));
    }

    if (layers?.volumeBar && volumeData.length >= 2) {
      const volumeSeries = chart.addHistogramSeries({
        color: "rgba(100, 116, 139, 0.24)",
        priceFormat: {
          type: "volume",
        },
        priceScaleId: "volume",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: {
          top: 0.82,
          bottom: 0,
        },
      });
      volumeSeries.setData(volumeData);
    }

    if (layers?.aiBand && upperBandHistoryData.length >= 2 && lowerBandHistoryData.length >= 2) {
      const upperBandHistorySeries = chart.addLineSeries({
        color: "rgba(30, 64, 175, 0.54)",
        lineWidth: 2,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      upperBandHistorySeries.setData(upperBandHistoryData);

      const lowerBandHistorySeries = chart.addLineSeries({
        color: "rgba(30, 64, 175, 0.54)",
        lineWidth: 2,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      lowerBandHistorySeries.setData(lowerBandHistoryData);
    }

    if (layers?.conservativeLine && conservativeHistoryData.length >= 2) {
      const conservativeHistorySeries = chart.addLineSeries({
        color: "rgba(4, 120, 87, 0.58)",
        lineWidth: 2,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      conservativeHistorySeries.setData(conservativeHistoryData);
    }

    if (upperBandData.length >= 2 && lowerBandData.length >= 2 && layers?.aiBand) {
      const upperBandSeries = chart.addLineSeries({
        color: "#172554",
        lineWidth: 4,
        lineStyle: LineStyle.Dashed,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      upperBandSeries.setData(upperBandData);

      const lowerBandSeries = chart.addLineSeries({
        color: "#172554",
        lineWidth: 4,
        lineStyle: LineStyle.Dashed,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      lowerBandSeries.setData(lowerBandData);
    }

    if (conservativeData.length >= 2 && layers?.conservativeLine) {
      const conservativeSeries = chart.addLineSeries({
        color: "#006b57",
        lineWidth: 4,
        lineStyle: LineStyle.Dashed,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
      });
      conservativeSeries.setData(conservativeData);
    }

    chart.timeScale().fitContent();
    applyInitialVisibleRange(chart, chartTimelineDates, timeframe);
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleRangeChange);
    handleVisibleRangeChange();
    requestAnimationFrame(updateForecastMarker);

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      chart.applyOptions({ width: entry.contentRect.width, height: Math.max(entry.contentRect.height, 520) });
      requestAnimationFrame(updateForecastMarker);
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleRangeChange);
      chart.remove();
    };
  }, [
    chartTimelineDates,
    data,
    timeframe,
    chartType,
    conservativeData,
    conservativeHistoryData,
    lowerBandData,
    lowerBandHistoryData,
    markerDate,
    onVisibleDatesChange,
    upperBandData,
    upperBandHistoryData,
    useSeparatePredictionScale,
    layers?.aiBand,
    layers?.conservativeLine,
    layers?.volumeBar,
    volumeData,
    whitespaceDates,
  ]);

  return (
    <div className="chart-frame">
      <div className="chart-frame__meta">
        <span>
          {ticker} / {timeframe}
        </span>
        <span>{chartType === "candles" ? "캔들" : "라인"}</span>
      </div>
      <div className="chart-frame__canvas-wrap">
        <div ref={containerRef} className="chart-frame__canvas" />
        <div ref={forecastMarkerRef} className="chart-forecast-marker" aria-hidden="true">
          <span title="과거 예측 이력과 최신 예측을 나누는 기준일">모델 기준일 경계</span>
        </div>
      </div>
      <div className="chart-legend">
        {conservativeData.length >= 2 ? (
          <span className="chart-legend__item chart-legend__item--latest-line" title="현재 모델 기준일 이후 예측">
            {latestForecastLabel}
          </span>
        ) : null}
        {upperBandData.length >= 2 && lowerBandData.length >= 2 ? (
          <span className="chart-legend__item chart-legend__item--latest-band" title="현재 모델 기준일 이후 AI 위험 범위">
            최신 AI 위험 범위
          </span>
        ) : null}
        {conservativeHistoryData.length >= 2 || upperBandHistoryData.length >= 2 ? (
          <span className="chart-legend__item chart-legend__item--history" title="과거 각 날짜에서 모델이 본 대표 horizon 기준값">
            과거 예측 이력
          </span>
        ) : null}
        {layers?.volumeBar && volumeData.length >= 2 ? <span className="chart-legend__item chart-legend__item--volume">거래량</span> : null}
        {markerDate ? <span className="chart-legend__item chart-legend__item--start">모델 기준일 경계</span> : null}
        {useSeparatePredictionScale ? <span className="chart-legend__muted">예측 범위가 넓어 별도 가격 축으로 표시 중</span> : null}
        {!canDrawBand && !canDrawConservativeLine && (prediction || bandPrediction) && timeframe !== "1M" ? (
          <span className="chart-legend__muted">{warning ?? "표시 가능한 예측선이 없습니다."}</span>
        ) : null}
      </div>
    </div>
  );
}
