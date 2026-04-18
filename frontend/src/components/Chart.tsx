"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi } from "lightweight-charts";
import { PriceBar } from "@/api/client";

interface ChartProps {
  data: PriceBar[];
  ticker: string;
}

export default function Chart({ data, ticker }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 400,
      layout: { background: { color: "#0f172a" }, textColor: "#94a3b8" },
      grid: { vertLines: { color: "#1e293b" }, horzLines: { color: "#1e293b" } },
    });
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    candleSeries.setData(
      data.map((d) => ({
        time: d.date,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }))
    );

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [data]);

  return (
    <div className="w-full rounded-lg overflow-hidden bg-slate-900 p-4">
      <h2 className="text-white text-lg font-semibold mb-2">{ticker}</h2>
      <div ref={containerRef} />
    </div>
  );
}
