export interface IndicatorChartPoint {
  date: string;
  value: number;
}

export interface IndicatorSummaryItem {
  label: string;
  value: string;
  tone?: "good" | "bad" | "neutral";
}

export interface IndicatorChartSeries {
  id: string;
  label: string;
  groupLabel: string;
  points: IndicatorChartPoint[];
  color?: string;
  baseline?: number;
  fixedRange?: {
    min: number;
    max: number;
  };
  latestLabel?: string;
  emptyMessage?: string;
  summary?: IndicatorSummaryItem[];
}

interface IndicatorPanelProps {
  series: IndicatorChartSeries[];
  emptyMessage?: string;
  note?: string | null;
  timelineDates?: string[];
}

const SVG_WIDTH = 640;
const SVG_HEIGHT = 112;
const PAD_X = 22;
const PAD_TOP = 12;
const PAD_BOTTOM = 18;

function formatAxis(value: number) {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

function getTimelineDates(series: IndicatorChartSeries, timelineDates?: string[]) {
  if (timelineDates && timelineDates.length > 0) {
    return timelineDates;
  }
  return series.points.map((point) => point.date);
}

function getVisibleValues(series: IndicatorChartSeries, timelineDates?: string[]) {
  const visibleDateSet = timelineDates && timelineDates.length > 0 ? new Set(timelineDates) : null;
  return series.points
    .filter((point) => !visibleDateSet || visibleDateSet.has(point.date))
    .map((point) => point.value)
    .filter(Number.isFinite);
}

function resolveRange(series: IndicatorChartSeries, timelineDates?: string[]) {
  const values = getVisibleValues(series, timelineDates);
  if (series.fixedRange) {
    return series.fixedRange;
  }
  if (series.baseline != null) {
    values.push(series.baseline);
  }
  if (values.length === 0) {
    return { min: 0, max: 1 };
  }
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const padding = Math.abs(min) > 1 ? Math.abs(min) * 0.08 : 0.5;
    min -= padding;
    max += padding;
  } else {
    const padding = (max - min) * 0.12;
    min -= padding;
    max += padding;
  }
  return { min, max };
}

function yForValue(value: number, min: number, max: number) {
  const plotHeight = SVG_HEIGHT - PAD_TOP - PAD_BOTTOM;
  const ratio = (value - min) / (max - min);
  return PAD_TOP + (1 - ratio) * plotHeight;
}

function buildPath(series: IndicatorChartSeries, timelineDates: string[], min: number, max: number) {
  const plotWidth = SVG_WIDTH - PAD_X * 2;
  const denominator = Math.max(timelineDates.length - 1, 1);
  const pointByDate = new Map(series.points.map((point) => [point.date, point.value]));
  let path = "";
  let segmentOpen = false;

  timelineDates.forEach((date, index) => {
    const value = pointByDate.get(date);
    if (typeof value !== "number" || !Number.isFinite(value)) {
      segmentOpen = false;
      return;
    }
    const x = PAD_X + (index / denominator) * plotWidth;
    const y = yForValue(value, min, max);
    path += `${segmentOpen ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)} `;
    segmentOpen = true;
  });

  return path.trim();
}

function getDateLabel(timelineDates: string[], position: "start" | "end") {
  if (timelineDates.length === 0) {
    return "-";
  }
  const date = position === "start" ? timelineDates[0] : timelineDates[timelineDates.length - 1];
  return date.slice(5);
}

function IndicatorChart({ series, timelineDates: inputTimelineDates }: { series: IndicatorChartSeries; timelineDates?: string[] }) {
  const timelineDates = getTimelineDates(series, inputTimelineDates);
  const visibleValues = getVisibleValues(series, timelineDates);
  const { min, max } = resolveRange(series, timelineDates);
  const baselineVisible = series.baseline != null && series.baseline >= min && series.baseline <= max;
  const baselineY = baselineVisible ? yForValue(series.baseline as number, min, max) : null;
  const midY = yForValue((min + max) / 2, min, max);
  const color = series.color ?? "#1f2937";

  if (visibleValues.length < 2) {
    return <div className="indicator-card__empty">{series.emptyMessage ?? "표시할 지표 데이터가 없습니다."}</div>;
  }

  return (
    <svg className="indicator-card__svg" viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} role="img" aria-label={series.label}>
      <line x1={PAD_X} x2={SVG_WIDTH - PAD_X} y1={PAD_TOP} y2={PAD_TOP} className="indicator-grid-line" />
      <line x1={PAD_X} x2={SVG_WIDTH - PAD_X} y1={midY} y2={midY} className="indicator-grid-line" />
      <line
        x1={PAD_X}
        x2={SVG_WIDTH - PAD_X}
        y1={SVG_HEIGHT - PAD_BOTTOM}
        y2={SVG_HEIGHT - PAD_BOTTOM}
        className="indicator-grid-line"
      />
      {baselineY != null ? (
        <line x1={PAD_X} x2={SVG_WIDTH - PAD_X} y1={baselineY} y2={baselineY} className="indicator-baseline" />
      ) : null}
      <path d={buildPath(series, timelineDates, min, max)} fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      <text x={PAD_X} y={10} className="indicator-axis-label">
        {formatAxis(max)}
      </text>
      <text x={PAD_X} y={SVG_HEIGHT - 3} className="indicator-axis-label">
        {formatAxis(min)}
      </text>
      <text x={PAD_X} y={SVG_HEIGHT - 3} className="indicator-date-label">
        {getDateLabel(timelineDates, "start")}
      </text>
      <text x={SVG_WIDTH - PAD_X} y={SVG_HEIGHT - 3} textAnchor="end" className="indicator-date-label">
        {getDateLabel(timelineDates, "end")}
      </text>
    </svg>
  );
}

export default function IndicatorPanel({ series, emptyMessage, note, timelineDates }: IndicatorPanelProps) {
  return (
    <section className="indicator-panel" aria-label="보조지표 패널">
      <div className="indicator-panel__top">
        <div>
          <span className="eyebrow">보조지표</span>
          <h3>선택 지표</h3>
        </div>
      </div>

      {note ? <div className="compact-note compact-note--warning">{note}</div> : null}

      {series.length === 0 ? (
        <div className="indicator-panel__empty">{emptyMessage ?? "오른쪽 패널에서 표시할 지표를 선택하세요."}</div>
      ) : (
        <div className="indicator-panel__list">
          {series.map((item) => (
            <article key={item.id} className="indicator-card">
              <div className="indicator-card__header">
                <div>
                  <span>{item.groupLabel}</span>
                  <strong>{item.label}</strong>
                </div>
                <em>{item.latestLabel ?? "-"}</em>
              </div>
              <IndicatorChart series={item} timelineDates={timelineDates} />
              {item.summary && item.summary.length > 0 ? (
                <div className="indicator-card__summary">
                  {item.summary.map((summary) => (
                    <div key={`${item.id}-${summary.label}`} className={`indicator-summary indicator-summary--${summary.tone ?? "neutral"}`}>
                      <span>{summary.label}</span>
                      <strong>{summary.value}</strong>
                    </div>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
