import { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "good" | "bad" | "neutral";
}

export default function MetricCard({ label, value, hint, tone = "default" }: MetricCardProps) {
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">{value}</div>
      {hint ? <div className="metric-card__hint">{hint}</div> : null}
    </article>
  );
}
