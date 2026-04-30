"use client";

import { useEffect, useMemo, useRef } from "react";
import { createChart, IChartApi, LineStyle } from "lightweight-charts";

import { PredictionResult, PriceBar } from "@/api/client";

interface ChartProps {
  data: PriceBar[];
  ticker: string;
  timeframe: "1D" | "1W" | "1M";
  chartType: "candles" | "line";
  prediction?: PredictionResult | null;
  layers?: {
    aiBand: boolean;
    conservativeLine: boolean;
  };
  timelineDates?: string[];
  onVisibleDatesChange?: (dates: string[]) => void;
}

interface OverlayPoint {
  time: string;
  value: number;
}

interface OverlayState {
  canDrawBand: boolean;
  canDrawConservativeLine: boolean;
  conservativeData: OverlayPoint[];
  upperBandData: OverlayPoint[];
  lowerBandData: OverlayPoint[];
  forecastStartDate: string | null;
  warning: string | null;
}

const PREDICTION_SCALE_ID = "prediction-overlay";

function buildOverlayData(dates: string[], values: number[]): OverlayPoint[] {
  if (dates.length === 0 || dates.length !== values.length) {
    return [];
  }

  const points = dates.map((date, index) => ({
    time: date,
    value: values[index],
  }));

  if (points.some((point) => !point.time || !Number.isFinite(point.value))) {
    return [];
  }

  return points;
}

function hasSameLength(dates: string[], values: number[] | null | undefined) {
  return Boolean(values && dates.length > 0 && dates.length === values.length);
}

function buildOverlayState(
  prediction: PredictionResult | null | undefined,
  layers: ChartProps["layers"],
  timeframe: ChartProps["timeframe"]
): OverlayState {
  if (!prediction || timeframe === "1M") {
    return {
      canDrawBand: false,
      canDrawConservativeLine: false,
      conservativeData: [],
      upperBandData: [],
      lowerBandData: [],
      forecastStartDate: null,
      warning: null,
    };
  }

  const forecastDates = prediction.forecast_dates ?? [];
  const lineValues = prediction.line_series?.length ? prediction.line_series : prediction.conservative_series;
  const hasLine = hasSameLength(forecastDates, lineValues);
  const hasUpper = hasSameLength(forecastDates, prediction.upper_band_series);
  const hasLower = hasSameLength(forecastDates, prediction.lower_band_series);

  if (!hasLine || !hasUpper || !hasLower) {
    return {
      canDrawBand: false,
      canDrawConservativeLine: false,
      conservativeData: [],
      upperBandData: [],
      lowerBandData: [],
      forecastStartDate: null,
      warning: "예측 날짜와 시리즈 길이가 맞지 않아 예측선을 숨겼습니다.",
    };
  }

  const conservativeData = buildOverlayData(forecastDates, lineValues);
  const upperBandData = buildOverlayData(forecastDates, prediction.upper_band_series);
  const lowerBandData = buildOverlayData(forecastDates, prediction.lower_band_series);
  const hasInvalidValue =
    conservativeData.length !== forecastDates.length ||
    upperBandData.length !== forecastDates.length ||
    lowerBandData.length !== forecastDates.length;

  if (hasInvalidValue) {
    return {
      canDrawBand: false,
      canDrawConservativeLine: false,
      conservativeData: [],
      upperBandData: [],
      lowerBandData: [],
      forecastStartDate: null,
      warning: "예측 데이터에 표시할 수 없는 값이 있어 예측선을 숨겼습니다.",
    };
  }

  return {
    canDrawBand: Boolean(layers?.aiBand) && upperBandData.length >= 2 && lowerBandData.length >= 2,
    canDrawConservativeLine: Boolean(layers?.conservativeLine) && conservativeData.length >= 2,
    conservativeData,
    upperBandData,
    lowerBandData,
    forecastStartDate: forecastDates[0] ?? null,
    warning: null,
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

export default function Chart({ data, ticker, timeframe, chartType, prediction, layers, timelineDates, onVisibleDatesChange }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const forecastMarkerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const overlayState = useMemo(() => buildOverlayState(prediction, layers, timeframe), [prediction, layers, timeframe]);
  const {
    canDrawBand,
    canDrawConservativeLine,
    conservativeData,
    forecastStartDate,
    lowerBandData,
    upperBandData,
    warning,
  } = overlayState;
  const chartTimelineDates = useMemo(() => {
    if (timelineDates && timelineDates.length > 0) {
      return timelineDates;
    }
    return data.map((item) => item.date);
  }, [data, timelineDates]);
  const priceDateSet = useMemo(() => new Set(data.map((item) => item.date)), [data]);
  const whitespaceDates = useMemo(
    () => chartTimelineDates.filter((date) => !priceDateSet.has(date)).map((time) => ({ time })),
    [chartTimelineDates, priceDateSet]
  );
  const useSeparatePredictionScale = useMemo(
    () => shouldUseSeparatePredictionScale(data, [...conservativeData, ...upperBandData, ...lowerBandData]),
    [conservativeData, data, lowerBandData, upperBandData]
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
    chartRef.current = chart;
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
      if (!forecastStartDate) {
        marker.style.display = "none";
        return;
      }

      const coordinate = chart.timeScale().timeToCoordinate(forecastStartDate);
      if (coordinate == null) {
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

      candleSeries.setData(
        [
          ...data.map((item) => ({
            time: item.date,
            open: item.open ?? item.close,
            high: item.high ?? item.close,
            low: item.low ?? item.close,
            close: item.close,
          })),
          ...whitespaceDates,
        ]
      );
    } else {
      const lineSeries = chart.addLineSeries({
        color: "#1f2937",
        lineWidth: 2,
      });
      lineSeries.setData(
        [
          ...data.map((item) => ({
            time: item.date,
            value: item.close,
          })),
          ...whitespaceDates,
        ]
      );
    }

    if (canDrawBand) {
      const upperBandSeries = chart.addLineSeries({
        color: "#4b5563",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      upperBandSeries.setData(upperBandData);

      const lowerBandSeries = chart.addLineSeries({
        color: "#4b5563",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      lowerBandSeries.setData(lowerBandData);
    }

    if (canDrawConservativeLine) {
      const conservativeSeries = chart.addLineSeries({
        color: "#111827",
        lineWidth: 2,
        lineStyle: LineStyle.Dotted,
        priceScaleId: useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right",
        priceLineVisible: false,
      });
      conservativeSeries.setData(conservativeData);
    }

    chart.timeScale().fitContent();
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
    canDrawBand,
    canDrawConservativeLine,
    conservativeData,
    forecastStartDate,
    lowerBandData,
    onVisibleDatesChange,
    upperBandData,
    useSeparatePredictionScale,
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
          <span>예측 시작</span>
        </div>
      </div>
      <div className="chart-legend">
        {canDrawBand ? <span className="chart-legend__item chart-legend__item--band">AI 밴드 상단/하단</span> : null}
        {canDrawConservativeLine ? (
          <span className="chart-legend__item chart-legend__item--line">보수적 예측</span>
        ) : null}
        {forecastStartDate ? <span className="chart-legend__item chart-legend__item--start">예측 시작 기준선</span> : null}
        {useSeparatePredictionScale ? <span className="chart-legend__muted">예측 밴드가 넓어 별도 scale로 표시 중</span> : null}
        {!canDrawBand && !canDrawConservativeLine && prediction && timeframe !== "1M" ? (
          <span className="chart-legend__muted">{warning ?? "표시 가능한 예측선이 없습니다."}</span>
        ) : null}
      </div>
    </div>
  );
}
