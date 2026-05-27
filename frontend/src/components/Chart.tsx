"use client";

import { useEffect, useMemo, useRef } from "react";
import { createChart, LineStyle } from "lightweight-charts";

import {
  PredictionResult,
  PriceBar,
  ProductBandHistoryPoint,
  ProductLineHistoryPoint,
} from "@/api/client";
import { PREDICTION_SCALE_ID } from "@/lib/chart/types";
import {
  applyInitialVisibleRange,
  buildCandleData,
  buildOverlayState,
  buildPriceLineData,
  buildVolumeData,
  emitVisibleDates,
  getLatestPriceDate,
  shouldUseSeparatePredictionScale,
  sortUniqueByTime,
} from "@/lib/chart/utils";

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
    () =>
      buildOverlayState(
        prediction,
        bandPrediction,
        predictionHistory,
        bandPredictionHistory,
        layers,
        timeframe,
        latestPriceDate
      ),
    [
      bandPrediction,
      bandPredictionHistory,
      latestPriceDate,
      prediction,
      predictionHistory,
      layers,
      timeframe,
    ]
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
  const markerDate = timeframe !== "1M" && layers?.aiBand && modelMarkerDate ? modelMarkerDate : null;
  const latestForecastLabel =
    timeframe === "1W" ? `보수적 기준선 (${Math.max(conservativeData.length, 1)}주)` : "보수적 기준선";

  const chartTimelineDates = useMemo(() => {
    const source = timelineDates && timelineDates.length > 0 ? timelineDates : data.map((item) => item.date);
    const dates = markerDate ? [...source, markerDate] : source;
    return sortUniqueByTime(dates.map((time) => ({ time }))).map((item) => item.time);
  }, [data, timelineDates, markerDate]);

  const priceDateSet = useMemo(
    () => new Set(sortUniqueByTime(data.map((item) => ({ time: item.date }))).map((item) => item.time)),
    [data]
  );
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
      rightPriceScale: { borderColor: "rgba(17, 24, 39, 0.12)" },
      timeScale: { borderColor: "rgba(17, 24, 39, 0.12)" },
      crosshair: { mode: 1 },
    });
    if (useSeparatePredictionScale) {
      chart.priceScale(PREDICTION_SCALE_ID).applyOptions({
        scaleMargins: { top: 0.08, bottom: 0.08 },
      });
    }

    const updateForecastMarker = () => {
      const marker = forecastMarkerRef.current;
      if (!marker) return;
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
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 },
      });
      volumeSeries.setData(volumeData);
    }

    const overlayScaleId = useSeparatePredictionScale ? PREDICTION_SCALE_ID : "right";

    if (layers?.aiBand && upperBandHistoryData.length >= 2 && lowerBandHistoryData.length >= 2) {
      const upperBandHistorySeries = chart.addLineSeries({
        color: "rgba(29, 78, 216, 0.26)",
        lineWidth: 2,
        priceScaleId: overlayScaleId,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      upperBandHistorySeries.setData(upperBandHistoryData);

      const lowerBandHistorySeries = chart.addLineSeries({
        color: "rgba(29, 78, 216, 0.26)",
        lineWidth: 2,
        priceScaleId: overlayScaleId,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      lowerBandHistorySeries.setData(lowerBandHistoryData);
    }

    if (layers?.conservativeLine && conservativeHistoryData.length >= 2) {
      const conservativeHistorySeries = chart.addLineSeries({
        color: "rgba(4, 120, 87, 0.28)",
        lineWidth: 2,
        priceScaleId: overlayScaleId,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      conservativeHistorySeries.setData(conservativeHistoryData);
    }

    // 단일 예측점을 라인 마커로 그리면 확대 중 캔버스가 깨질 수 있어 선은 2점 이상일 때만 그린다.
    if (upperBandData.length >= 2 && lowerBandData.length >= 2 && layers?.aiBand) {
      const upperBandSeries = chart.addLineSeries({
        color: "#1d4ed8",
        lineWidth: 4,
        lineStyle: LineStyle.Dashed,
        priceScaleId: overlayScaleId,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      upperBandSeries.setData(upperBandData);

      const lowerBandSeries = chart.addLineSeries({
        color: "#1d4ed8",
        lineWidth: 4,
        lineStyle: LineStyle.Dashed,
        priceScaleId: overlayScaleId,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      lowerBandSeries.setData(lowerBandData);
    }

    if (conservativeData.length >= 2 && layers?.conservativeLine) {
      const conservativeSeries = chart.addLineSeries({
        color: "#047857",
        lineWidth: 4,
        lineStyle: LineStyle.Dashed,
        priceScaleId: overlayScaleId,
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
      if (!entry) return;
      chart.applyOptions({
        width: entry.contentRect.width,
        height: Math.max(entry.contentRect.height, 520),
      });
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
          <span title="자동 갱신 AI 밴드의 마지막 저장 기준일">밴드 기준일</span>
        </div>
      </div>
      <div className="chart-legend">
        {conservativeData.length >= 2 ? (
          <span
            className="chart-legend__item chart-legend__item--latest-line"
            title="safe_line_score를 asof 종가에 곱해 가격으로 환산한 보수적 기준선"
          >
            {latestForecastLabel}
          </span>
        ) : null}
        {upperBandData.length >= 2 && lowerBandData.length >= 2 ? (
          <span
            className="chart-legend__item chart-legend__item--latest-band"
            title="밴드 기준일 이후 AI 위험 범위"
          >
            최신 AI 위험 범위
          </span>
        ) : null}
        {conservativeHistoryData.length >= 2 || upperBandHistoryData.length >= 2 ? (
          <span
            className="chart-legend__item chart-legend__item--history"
            title="과거 각 날짜에서 모델이 본 대표 horizon 기준값"
          >
            과거 예측 이력
          </span>
        ) : null}
        {layers?.volumeBar && volumeData.length >= 2 ? (
          <span className="chart-legend__item chart-legend__item--volume">거래량</span>
        ) : null}
        {markerDate ? (
          <span className="chart-legend__item chart-legend__item--start">밴드 기준일</span>
        ) : null}
        {useSeparatePredictionScale ? (
          <span className="chart-legend__muted">예측 범위가 넓어 별도 가격 축으로 표시 중</span>
        ) : null}
        {!canDrawBand && !canDrawConservativeLine && (prediction || bandPrediction) && timeframe !== "1M" ? (
          <span className="chart-legend__muted">{warning ?? "표시 가능한 예측선이 없습니다."}</span>
        ) : null}
      </div>
    </div>
  );
}
