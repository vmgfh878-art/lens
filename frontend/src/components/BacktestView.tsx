"use client";

import { useEffect, useMemo, useState } from "react";

import { AiRunDetail, BacktestSummary, DisplayTimeframe, fetchAiRun, fetchAiRuns, fetchRunBacktests } from "@/api/client";
import MetricCard from "@/components/MetricCard";

const STRATEGIES = ["band_breakout_v1"];
const TIMEFRAMES: DisplayTimeframe[] = ["1D", "1W", "1M"];
const DEMO_RUN_MODELS = new Set(["line_band_composite", "patchtst"]);

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${formatNumber(value)}%`;
}

function getMetaNumber(meta: Record<string, unknown>, key: string) {
  const value = meta[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function extractErrorMessage(error: unknown) {
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. 127.0.0.1:8000 서버가 켜져 있는지 확인해주세요.";
    }
    return error.message;
  }
  return "백테스트 결과를 불러오지 못했습니다.";
}

function getMetricNumber(metrics: Record<string, unknown> | undefined, key: string) {
  const value = metrics?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

const QUALITY_METRICS = [
  { key: "coverage", label: "커버리지" },
  { key: "avg_band_width", label: "평균 밴드 폭" },
  { key: "mae", label: "MAE" },
  { key: "smape", label: "SMAPE" },
  { key: "spearman_ic", label: "Spearman IC" },
  { key: "top_k_long_spread", label: "상위 롱 스프레드" },
  { key: "long_short_spread", label: "롱/숏 스프레드" },
  { key: "fee_adjusted_return", label: "수수료 반영 수익" },
  { key: "fee_adjusted_sharpe", label: "수수료 반영 샤프" },
  { key: "fee_adjusted_turnover", label: "수수료 반영 회전율" },
  { key: "direction_accuracy", label: "방향 정확도" },
];

export default function BacktestView() {
  const [strategyName, setStrategyName] = useState(STRATEGIES[0]);
  const [timeframe, setTimeframe] = useState<DisplayTimeframe>("1D");
  const [runId, setRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<AiRunDetail | null>(null);
  const [backtests, setBacktests] = useState<BacktestSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function loadBacktests(nextStrategy = strategyName, nextTimeframe = timeframe) {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const runsResponse = await fetchAiRuns({
        status: "completed",
        modelName: "",
        timeframe: nextTimeframe,
        limit: 50,
      });
      const candidateRuns = runsResponse.data.filter((run) => run.model_name && DEMO_RUN_MODELS.has(run.model_name));
      const latestRun = candidateRuns[0] ?? null;

      if (!latestRun) {
        setRunId(null);
        setBacktests([]);
        setRunDetail(null);
        return;
      }

      let selectedRun = latestRun;
      let selectedBacktests: BacktestSummary[] = [];

      for (const run of candidateRuns) {
        const backtestsResponse = await fetchRunBacktests(run.run_id, {
          strategyName: nextStrategy,
          timeframe: nextTimeframe,
          limit: 50,
        });
        if (backtestsResponse.data.length > 0) {
          selectedRun = run;
          selectedBacktests = backtestsResponse.data;
          break;
        }
      }

      setRunId(selectedRun.run_id);
      const runResponse = await fetchAiRun(selectedRun.run_id, { includeConfig: false });
      setRunDetail(runResponse.data);
      setBacktests(selectedBacktests);
    } catch (error) {
      setBacktests([]);
      setRunDetail(null);
      setErrorMessage(extractErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadBacktests();
  }, []);

  const selectedBacktest = useMemo(() => backtests[0] ?? null, [backtests]);
  const feeBps = selectedBacktest ? getMetaNumber(selectedBacktest.meta, "fee_bps") : null;
  const feeAdjustedReturn = selectedBacktest?.fee_adjusted_return_pct ?? selectedBacktest?.return_pct ?? null;
  const feeAdjustedSharpe = selectedBacktest?.fee_adjusted_sharpe ?? selectedBacktest?.sharpe ?? null;
  const avgTurnover = selectedBacktest?.avg_turnover ?? (selectedBacktest ? getMetaNumber(selectedBacktest.meta, "avg_turnover") : null);
  const grossReturnPct = selectedBacktest ? getMetaNumber(selectedBacktest.meta, "gross_return_pct") : null;
  const grossSharpe = selectedBacktest ? getMetaNumber(selectedBacktest.meta, "gross_sharpe") : null;

  return (
    <div className="view-stack">
      <header className="view-header">
        <div>
          <div className="eyebrow">백테스트</div>
          <h1>전략 결과</h1>
        </div>
        <div className="status-badge status-badge--neutral">{runId ?? "run 없음"}</div>
      </header>

      <section className="panel">
        <div className="toolbar">
          <label className="select-field">
            <span>전략</span>
            <select
              value={strategyName}
              onChange={(event) => {
                const next = event.target.value;
                setStrategyName(next);
                void loadBacktests(next, timeframe);
              }}
            >
              {STRATEGIES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="select-field">
            <span>차트 단위</span>
            <select
              value={timeframe}
              onChange={(event) => {
                const next = event.target.value as DisplayTimeframe;
                setTimeframe(next);
                void loadBacktests(strategyName, next);
              }}
            >
              {TIMEFRAMES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <MetricCard label="수수료(bp)" value={feeBps == null ? "-" : formatNumber(feeBps, 1)} />
        </div>
      </section>

      {errorMessage ? <div className="notice notice--error">{errorMessage}</div> : null}

      <section className="metric-grid metric-grid--six">
        <MetricCard label="수수료 반영 수익률" value={formatPercent(feeAdjustedReturn)} tone="good" />
        <MetricCard label="수수료 반영 샤프" value={formatNumber(feeAdjustedSharpe)} />
        <MetricCard label="최대낙폭" value={formatPercent(selectedBacktest?.mdd)} tone="bad" />
        <MetricCard label="승률" value={formatPercent(selectedBacktest?.win_rate)} />
        <MetricCard label="평균 회전율" value={formatNumber(avgTurnover)} />
        <MetricCard label="거래 수" value={formatNumber(selectedBacktest?.num_trades, 0)} />
      </section>

      <section className="metric-grid metric-grid--four">
        <MetricCard label="수수료(bp)" value={feeBps == null ? "-" : formatNumber(feeBps, 1)} />
        <MetricCard label="수수료 전 수익률" value={formatPercent(grossReturnPct)} />
        <MetricCard label="수수료 전 샤프" value={formatNumber(grossSharpe)} />
        <MetricCard label="수익 팩터" value={formatNumber(selectedBacktest?.profit_factor)} />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div className="eyebrow">평가 지표</div>
          <h2>선택 단위 최신 완료 run 품질</h2>
        </div>
        {runDetail ? (
          <div className="quality-grid">
            <div>
              <h3>검증 지표</h3>
              <div className="metric-grid metric-grid--quality">
                {QUALITY_METRICS.map((metric) => (
                  <MetricCard
                    key={`val-${metric.key}`}
                    label={metric.label}
                    value={formatNumber(getMetricNumber(runDetail.val_metrics, metric.key), 4)}
                  />
                ))}
              </div>
            </div>
            <div>
              <h3>테스트 지표</h3>
              <div className="metric-grid metric-grid--quality">
                {QUALITY_METRICS.map((metric) => (
                  <MetricCard
                    key={`test-${metric.key}`}
                    label={metric.label}
                    value={formatNumber(getMetricNumber(runDetail.test_metrics, metric.key), 4)}
                  />
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-state">선택한 차트 단위의 완료 run이 없어서 평가 지표를 표시할 수 없습니다.</div>
        )}
      </section>

      <section className="chart-grid">
        <div className="panel placeholder-chart">
          <div className="panel-heading">
            <div className="eyebrow">수익 곡선</div>
            <h2>수익 곡선</h2>
          </div>
          <div className="empty-state">{isLoading ? "조회 중입니다." : "저장된 곡선 데이터가 있으면 여기에 표시합니다."}</div>
        </div>
        <div className="panel placeholder-chart">
          <div className="panel-heading">
            <div className="eyebrow">낙폭 차트</div>
            <h2>낙폭 차트</h2>
          </div>
          <div className="empty-state">{isLoading ? "조회 중입니다." : "저장된 낙폭 데이터가 있으면 여기에 표시합니다."}</div>
        </div>
      </section>

      {!isLoading && !selectedBacktest ? (
        <div className="empty-state">최근 완료 run에 저장된 백테스트 결과가 없습니다.</div>
      ) : null}
    </div>
  );
}
