// 백테스트 화면 표현 컴포넌트. SVG로 가격/누적수익 선과 매수/매도 마커 + 포지션 strip 표시.
// 부작용 없는 순수 표현. 부모(BacktestView)가 이미 client 컴포넌트라 "use client" 불필요.

import type { BacktestPoint, LineSeries, TradeEvent } from "@/lib/backtest/types";

function buildPath(values: Array<{ x: number; y: number }>) {
  return values.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
}

export function MiniLineChart({
  title,
  lines,
  markers,
}: {
  title: string;
  lines: LineSeries[];
  markers?: TradeEvent[];
}) {
  const width = 720;
  const height = 220;
  const padding = { top: 18, right: 18, bottom: 28, left: 42 };
  const allValues = lines.flatMap((line) => line.values.map((point) => point.value).filter(Number.isFinite));
  const longest = lines.reduce((current, line) => (line.values.length > current.length ? line.values : current), [] as LineSeries["values"]);

  if (allValues.length < 2 || longest.length < 2) {
    return <div className="empty-state">{title}를 표시할 데이터가 없습니다.</div>;
  }

  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const span = Math.max(maxValue - minValue, Math.abs(maxValue) * 0.02, 1);
  const xForIndex = (index: number, length: number) =>
    padding.left + (index / Math.max(length - 1, 1)) * (width - padding.left - padding.right);
  const yForValue = (value: number) => padding.top + ((maxValue - value) / span) * (height - padding.top - padding.bottom);

  return (
    <div className="risk-chart">
      <div className="risk-chart__header">
        <strong>{title}</strong>
        <span>
          {longest[0]?.date} - {longest[longest.length - 1]?.date}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
        {lines.map((line) => {
          const points = line.values.map((point, index) => ({
            x: xForIndex(index, line.values.length),
            y: yForValue(point.value),
          }));
          return <path key={line.label} d={buildPath(points)} stroke={line.color} />;
        })}
        {markers?.map((marker) => {
          const index = longest.findIndex((point) => point.date >= marker.date);
          if (index < 0) {
            return null;
          }
          const x = xForIndex(index, longest.length);
          const priceLine = lines[0];
          const markerPoint = priceLine?.values[index];
          if (!markerPoint) {
            return null;
          }
          return (
            <circle
              key={`${marker.kind}-${marker.date}-${marker.price}`}
              cx={x}
              cy={yForValue(markerPoint.value)}
              r="4"
              className={marker.kind === "entry" ? "risk-chart__marker-entry" : "risk-chart__marker-exit"}
            />
          );
        })}
      </svg>
      <div className="risk-chart__legend">
        {lines.map((line) => (
          <span key={line.label}>
            <i style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
        {markers && markers.length > 0 ? <span>마커: 매수/매도</span> : null}
      </div>
    </div>
  );
}

export function PositionStrip({ points }: { points: BacktestPoint[] }) {
  const sampled = points.length > 120 ? points.filter((_, index) => index % Math.ceil(points.length / 120) === 0) : points;
  return (
    <div className="position-strip" aria-label="보유와 현금 구간">
      {sampled.map((point) => (
        <span key={point.date} className={point.position === 1 ? "is-hold" : "is-cash"} title={`${point.date} ${point.position === 1 ? "보유" : "현금"}`} />
      ))}
    </div>
  );
}
