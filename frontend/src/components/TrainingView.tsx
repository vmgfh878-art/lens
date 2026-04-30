"use client";

import { Fragment, useEffect, useState } from "react";

import {
  AiRunDetail,
  AiRunStatus,
  AiRunSummary,
  EvaluationSummary,
  fetchAiRun,
  fetchAiRuns,
  fetchRunEvaluations,
} from "@/api/client";
import MetricCard from "@/components/MetricCard";

function formatValue(value: unknown, digits = 4) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "-";
    }
    return new Intl.NumberFormat("ko-KR", {
      maximumFractionDigits: digits,
    }).format(value);
  }
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return "-";
}

function getMetric(metrics: Record<string, unknown>, key: string) {
  return metrics[key] ?? null;
}

function extractErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. 127.0.0.1:8000 서버가 켜져 있는지 확인해주세요.";
    }
    return error.message;
  }
  return fallback;
}

const CONFIG_KEYS = ["seq_len", "patch_len", "stride", "d_model", "n_heads", "n_layers", "dropout", "lr", "epochs", "seed"];
const COMPOSITE_CONFIG_KEYS = [
  "line_model_run_id",
  "band_model_run_id",
  "composition_policy",
  "band_calibration_method",
  "prediction_composition_version",
];
const TRAINING_RUN_MODELS = new Set(["patchtst", "line_band_composite"]);
const QUALITY_METRICS = [
  { key: "coverage", label: "coverage" },
  { key: "avg_band_width", label: "avg_band_width" },
  { key: "mae", label: "mae" },
  { key: "smape", label: "smape" },
  { key: "spearman_ic", label: "spearman_ic" },
  { key: "top_k_long_spread", label: "top_k_long_spread" },
  { key: "long_short_spread", label: "long_short_spread" },
  { key: "fee_adjusted_return", label: "fee_adjusted_return" },
  { key: "fee_adjusted_sharpe", label: "fee_adjusted_sharpe" },
  { key: "fee_adjusted_turnover", label: "fee_adjusted_turnover" },
  { key: "direction_accuracy", label: "direction_accuracy" },
];

function formatStatusLabel(status: string | null | undefined) {
  if (status === "completed") {
    return "완료(completed)";
  }
  if (status === "failed_nan") {
    return "NaN 실패(failed_nan)";
  }
  if (status === "failed_quality_gate") {
    return "품질 게이트 실패(failed_quality_gate)";
  }
  return status ?? "-";
}

function isCompositeRun(detail: AiRunDetail | null) {
  return detail?.model_name === "line_band_composite" || detail?.config_summary?.line_model_run_id != null;
}

export default function TrainingView() {
  const [status, setStatus] = useState<AiRunStatus>("completed");
  const [runs, setRuns] = useState<AiRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<AiRunDetail | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function loadRuns(nextStatus = status) {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const response = await fetchAiRuns({ status: nextStatus, modelName: "", limit: 50 });
      const filteredRuns = response.data.filter((run) => TRAINING_RUN_MODELS.has(String(run.model_name ?? "")));
      setRuns(filteredRuns);
      const firstRunId = filteredRuns[0]?.run_id ?? null;
      setSelectedRunId(firstRunId);
      if (firstRunId) {
        await loadRunDetail(firstRunId);
      } else {
        setDetail(null);
        setEvaluations([]);
      }
    } catch (error) {
      setRuns([]);
      setSelectedRunId(null);
      setDetail(null);
      setEvaluations([]);
      setErrorMessage(extractErrorMessage(error, "run 목록을 불러오지 못했습니다."));
    } finally {
      setIsLoading(false);
    }
  }

  async function loadRunDetail(runId: string) {
    try {
      setSelectedRunId(runId);
      setErrorMessage(null);
      const detailResponse = await fetchAiRun(runId, { includeConfig: false });
      setDetail(detailResponse.data);
      const evaluationsResponse = await fetchRunEvaluations(runId, { limit: 8 });
      setEvaluations(evaluationsResponse.data);
    } catch (error) {
      setDetail(null);
      setEvaluations([]);
      setErrorMessage(extractErrorMessage(error, "run 상세를 불러오지 못했습니다."));
    }
  }

  useEffect(() => {
    void loadRuns("completed");
  }, []);

  return (
    <div className="view-stack">
      <header className="view-header">
        <div>
          <div className="eyebrow">모델 학습</div>
          <h1>run 콘솔</h1>
        </div>
        <div className="status-tabs">
          <button
            type="button"
            className={status === "completed" ? "is-active" : ""}
            onClick={() => {
              setStatus("completed");
              void loadRuns("completed");
            }}
          >
            완료(completed)
          </button>
          <button
            type="button"
            className={status === "failed_nan" ? "is-active" : ""}
            onClick={() => {
              setStatus("failed_nan");
              void loadRuns("failed_nan");
            }}
          >
            NaN 실패(failed_nan)
          </button>
          <button
            type="button"
            className={status === "failed_quality_gate" ? "is-active" : ""}
            onClick={() => {
              setStatus("failed_quality_gate");
              void loadRuns("failed_quality_gate");
            }}
          >
            품질 게이트 실패(failed_quality_gate)
          </button>
        </div>
      </header>

      {errorMessage ? <div className="notice notice--error">{errorMessage}</div> : null}

      <section className="training-layout">
        <div className="panel run-list-panel">
          <div className="panel-heading">
            <div className="eyebrow">run 목록</div>
            <h2>저장된 run</h2>
          </div>
          <div className="run-list">
            {runs.map((run) => (
              <button
                key={run.run_id}
                type="button"
                className={`run-row${selectedRunId === run.run_id ? " run-row--active" : ""}`}
                onClick={() => {
                  void loadRunDetail(run.run_id);
                }}
              >
                <span className="run-row__id">{run.run_id}</span>
                <span>{formatStatusLabel(run.status)}</span>
                <span>{run.model_name}</span>
                <span>{run.timeframe}</span>
                <span>{run.line_target_type ?? "-"}</span>
                <span>{formatValue(run.best_epoch, 0)}</span>
                <span>{formatValue(run.best_val_total)}</span>
                <span>{run.checkpoint_path ? "있음" : "없음"}</span>
              </button>
            ))}
            {!isLoading && runs.length === 0 ? <div className="empty-state">선택한 상태의 run이 없습니다.</div> : null}
          </div>
        </div>

        <div className="view-stack">
          <section className="metric-grid metric-grid--four">
            <MetricCard label="model_name" value={detail?.model_name ?? "-"} />
            <MetricCard label="timeframe" value={detail?.timeframe ?? "-"} />
            <MetricCard label="status" value={formatStatusLabel(detail?.status)} />
            <MetricCard label="best_epoch" value={formatValue(detail?.best_epoch, 0)} />
            <MetricCard label="best_val_total" value={formatValue(detail?.best_val_total)} />
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div className="eyebrow">config summary</div>
              <h2>설정 요약</h2>
            </div>
            {detail ? (
              <div className="config-grid">
                {CONFIG_KEYS.map((key) => (
                  <div key={key} className="config-item">
                    <span>{key}</span>
                    <strong>{formatValue(detail.config_summary[key])}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">run을 선택하면 설정 요약을 표시합니다.</div>
            )}
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div className="eyebrow">run provenance</div>
              <h2>조합 정보</h2>
            </div>
            {detail ? (
              <div className="provenance-grid provenance-grid--training">
                <span>run_id</span>
                <strong>{detail.run_id}</strong>
                <span>model_name</span>
                <strong>{detail.model_name ?? "-"}</strong>
                <span>feature_version</span>
                <strong>{detail.feature_version ?? "-"}</strong>
                {COMPOSITE_CONFIG_KEYS.map((key) => (
                  <Fragment key={key}>
                    <span key={`${key}-label`}>{key}</span>
                    <strong key={`${key}-value`}>{formatValue(detail.config_summary[key])}</strong>
                  </Fragment>
                ))}
              </div>
            ) : (
              <div className="empty-state">run을 선택하면 조합 정보를 표시합니다.</div>
            )}
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div className="eyebrow">summary</div>
              <h2>W&B / Optuna</h2>
            </div>
            <div className="summary-grid">
              <MetricCard label="wandb_run_id" value={detail?.wandb_run_id ?? "-"} />
              <MetricCard label="val mae" value={formatValue(detail ? getMetric(detail.val_metrics, "mae") : null)} />
              <MetricCard label="test mae" value={formatValue(detail ? getMetric(detail.test_metrics, "mae") : null)} />
              <MetricCard label="checkpoint" value={detail?.checkpoint_path ? "있음" : "없음"} />
            </div>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div className="eyebrow">line / band</div>
              <h2>모델별 지표 자리</h2>
            </div>
            {isCompositeRun(detail) ? (
              <div className="quality-grid">
                <div>
                  <h3>line 모델</h3>
                  <div className="summary-grid summary-grid--compact">
                    <MetricCard label="line_model_run_id" value={formatValue(detail?.config_summary.line_model_run_id)} />
                    <MetricCard label="model" value="PatchTST" />
                  </div>
                </div>
                <div>
                  <h3>band 모델</h3>
                  <div className="summary-grid summary-grid--compact">
                    <MetricCard label="band_model_run_id" value={formatValue(detail?.config_summary.band_model_run_id)} />
                    <MetricCard label="model" value="CNN-LSTM" />
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">composite run을 선택하면 line/band 모델 지표 자리를 표시합니다.</div>
            )}
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div className="eyebrow">품질 지표</div>
              <h2>val_metrics / test_metrics</h2>
            </div>
            {detail ? (
              <div className="quality-grid">
                <div>
                  <h3>val_metrics</h3>
                  <div className="metric-grid metric-grid--quality">
                    {QUALITY_METRICS.map((metric) => (
                      <MetricCard
                        key={`training-val-${metric.key}`}
                        label={metric.label}
                        value={formatValue(getMetric(detail.val_metrics, metric.key))}
                      />
                    ))}
                  </div>
                </div>
                <div>
                  <h3>test_metrics</h3>
                  <div className="metric-grid metric-grid--quality">
                    {QUALITY_METRICS.map((metric) => (
                      <MetricCard
                        key={`training-test-${metric.key}`}
                        label={metric.label}
                        value={formatValue(getMetric(detail.test_metrics, metric.key))}
                      />
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">run을 선택하면 val/test 품질 지표를 표시합니다.</div>
            )}
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div className="eyebrow">evaluations</div>
              <h2>평가 테이블</h2>
            </div>
            {evaluations.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>ticker</th>
                      <th>asof</th>
                      <th>coverage</th>
                      <th>mae</th>
                      <th>smape</th>
                      <th>direction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluations.map((item) => (
                      <tr key={`${item.run_id}-${item.ticker}-${item.asof_date}`}>
                        <td>{item.ticker ?? "-"}</td>
                        <td>{item.asof_date ?? "-"}</td>
                        <td>{formatValue(item.coverage)}</td>
                        <td>{formatValue(item.mae)}</td>
                        <td>{formatValue(item.smape)}</td>
                        <td>{formatValue(item.direction_accuracy)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">선택한 run의 평가 테이블이 비어 있습니다.</div>
            )}
          </section>
        </div>
      </section>
    </div>
  );
}
