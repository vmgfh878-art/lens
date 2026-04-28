"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi, LineStyle } from "lightweight-charts";

import { PredictionResult, PriceBar } from "@/api/client";

interface ChartProps {
  data: PriceBar[];
  ticker: string;
  timeframe: "1D" | "1W" | "1M";
  chartType: "candles" | "line";
  prediction?: PredictionResult | null;
  layers?: {
    indicators: boolean;
    aiBand: boolean;
    conservativeLine: boolean;
  };
}

interface OverlayPoint {
  time: string;
  value: number;
}

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

function isPriceScaleCompatible(data: PriceBar[], overlayData: OverlayPoint[]) {
  if (overlayData.length === 0) {
    return false;
  }

  const prices = data.flatMap((item) => [item.open ?? item.close, item.high ?? item.close, item.low ?? item.close, item.close]);
  const finitePrices = prices.filter((value) => Number.isFinite(value));
  if (finitePrices.length === 0) {
    return false;
  }

  const minPrice = Math.min(...finitePrices);
  const maxPrice = Math.max(...finitePrices);
  const minAllowed = minPrice > 0 ? minPrice * 0.2 : minPrice - Math.abs(maxPrice - minPrice) * 3;
  const maxAllowed = maxPrice > 0 ? maxPrice * 5 : maxPrice + Math.abs(maxPrice - minPrice) * 3;

  return overlayData.every((point) => point.value >= minAllowed && point.value <= maxAllowed);
}

export default function Chart({ data, ticker, timeframe, chartType, prediction, layers }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const lineValues = prediction?.line_series?.length ? prediction.line_series : prediction?.conservative_series ?? [];
  const conservativeData = prediction ? buildOverlayData(prediction.forecast_dates, lineValues) : [];
  const upperBandData = prediction ? buildOverlayData(prediction.forecast_dates, prediction.upper_band_series) : [];
  const lowerBandData = prediction ? buildOverlayData(prediction.forecast_dates, prediction.lower_band_series) : [];
  const canDrawConservativeLine =
    Boolean(layers?.conservativeLine) && isPriceScaleCompatible(data, conservativeData);
  const canDrawBand =
    Boolean(layers?.aiBand) &&
    isPriceScaleCompatible(data, upperBandData) &&
    isPriceScaleCompatible(data, lowerBandData);

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

    if (chartType === "candles") {
      const candleSeries = chart.addCandlestickSeries({
        upColor: "#0f9f6e",
        downColor: "#d92d20",
        wickUpColor: "#0f9f6e",
        wickDownColor: "#d92d20",
        borderVisible: false,
      });

      candleSeries.setData(
        data.map((item) => ({
          time: item.date,
          open: item.open ?? item.close,
          high: item.high ?? item.close,
          low: item.low ?? item.close,
          close: item.close,
        }))
      );
    } else {
      const lineSeries = chart.addLineSeries({
        color: "#1f2937",
        lineWidth: 2,
      });
      lineSeries.setData(
        data.map((item) => ({
          time: item.date,
          value: item.close,
        }))
      );
    }

    if (canDrawBand) {
      const upperBandSeries = chart.addLineSeries({
        color: "#4b5563",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      upperBandSeries.setData(upperBandData);

      const lowerBandSeries = chart.addLineSeries({
        color: "#4b5563",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      lowerBandSeries.setData(lowerBandData);
    }

    if (canDrawConservativeLine) {
      const conservativeSeries = chart.addLineSeries({
        color: "#111827",
        lineWidth: 2,
        priceLineVisible: false,
      });
      conservativeSeries.setData(conservativeData);
    }

    chart.timeScale().fitContent();

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      chart.applyOptions({ width: entry.contentRect.width, height: Math.max(entry.contentRect.height, 520) });
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [
    data,
    timeframe,
    chartType,
    canDrawBand,
    canDrawConservativeLine,
    conservativeData,
    lowerBandData,
    upperBandData,
  ]);

  return (
    <div className="chart-frame">
      <div className="chart-frame__meta">
        <span>
          {ticker} / {timeframe}
        </span>
        <span>{chartType === "candles" ? "캔들" : "라인"}</span>
      </div>
      <div ref={containerRef} className="chart-frame__canvas" />
      <div className="chart-legend">
        {canDrawBand ? <span className="chart-legend__item chart-legend__item--band">예측 밴드 상단/하단</span> : null}
        {canDrawConservativeLine ? (
          <span className="chart-legend__item chart-legend__item--line">보수적 예측선</span>
        ) : null}
        {!canDrawBand && !canDrawConservativeLine && prediction && timeframe !== "1M" ? (
          <span className="chart-legend__muted">표시 가능한 예측선이 없습니다.</span>
        ) : null}
      </div>
    </div>
  );
}
