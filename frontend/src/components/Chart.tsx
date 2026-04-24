"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi } from "lightweight-charts";

import { PriceBar } from "@/api/client";

interface ChartProps {
  data: PriceBar[];
  ticker: string;
  timeframe: "1D" | "1W" | "1M";
}

export default function Chart({ data, ticker, timeframe }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) {
      return;
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: "#fff8ef" },
        textColor: "#6b7280",
      },
      grid: {
        vertLines: { color: "rgba(146, 64, 14, 0.08)" },
        horzLines: { color: "rgba(146, 64, 14, 0.08)" },
      },
      rightPriceScale: {
        borderColor: "rgba(146, 64, 14, 0.16)",
      },
      timeScale: {
        borderColor: "rgba(146, 64, 14, 0.16)",
      },
    });
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#15803d",
      downColor: "#b91c1c",
      wickUpColor: "#15803d",
      wickDownColor: "#b91c1c",
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

    chart.timeScale().fitContent();

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      chart.applyOptions({ width: entry.contentRect.width });
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [data, timeframe]);

  return (
    <div>
      <div className="inline-note" style={{ marginTop: 0, marginBottom: "12px" }}>
        {ticker}의 {timeframe} 기준 캔들 차트입니다.
      </div>
      <div ref={containerRef} />
    </div>
  );
}
