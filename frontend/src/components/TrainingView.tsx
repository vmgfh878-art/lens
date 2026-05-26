"use client";

import { useEffect, useMemo, useState } from "react";

import { AiRunDetail, AiRunSummary, fetchAiRun, fetchAiRuns } from "@/api/client";
import { PRODUCT_RUN_IDS, PRODUCT_SLOTS as PRODUCT_SLOT_CONFIGS } from "@/lib/productSlots";
import type { ProductSlotId } from "@/lib/productSlots";

type ProductSlotKind = "line" | "band" | "preparing-line" | "preparing-band";
type ProductSlotStatus = "사용 중" | "데이터 확인 중" | "연결 필요" | "준비 중";
type ExperimentCategory = "previous" | "quality_failed";
type ExperimentKind = "line" | "band";
type SelectedItem =
  | { kind: "slot"; slotId: ProductSlotId }
  | { kind: "experiment"; runId: string; category: ExperimentCategory };

interface ProductSlot {
  id: ProductSlotId;
  kind: ProductSlotKind;
  title: string;
  model: string;
  version: string | null;
  timeframe: "1D" | "1W";
  runId: string | null;
  summary: string;
}

interface MetricDefinition {
  label: string;
  keys: string[];
  format?: "number" | "rate" | "pct_point";
}

interface ComparisonMetricDefinition extends MetricDefinition {
  id: string;
  better: "higher" | "lower" | "target_coverage" | "neutral";
}

interface ComparisonRow {
  id: string;
  label: string;
  productValue: number;
  experimentValue: number;
  productText: string;
  experimentText: string;
  diffText: string;
  interpretation: string;
  result: "better" | "worse" | "similar" | "neutral";
}

interface GoalCardProps {
  title: string;
  target: string;
  actual: string;
  diff: string;
  judgement: "통과" | "보통" | "개선 필요" | "준비 중" | "저장 없음";
  description: string;
  tone?: "good" | "neutral" | "warn";
}

interface GoalCardData extends GoalCardProps {
  id: string;
}

interface ExperimentListItem {
  run: AiRunSummary;
  detail: AiRunDetail;
  category: ExperimentCategory;
  kind: ExperimentKind;
  tag: string;
}

interface DetailField {
  key: string;
  label: string;
  value: string;
  monospace?: boolean;
}

const TRAINING_RUN_MODELS = new Set(["patchtst", "cnn_lstm", "tide", "line_band_composite"]);

const PRODUCT_SLOTS: ProductSlot[] = PRODUCT_SLOT_CONFIGS.map((slot) => ({
  id: slot.id,
  kind: slot.status === "deferred" ? (slot.kind === "line" ? "preparing-line" : "preparing-band") : slot.kind,
  title: slot.title,
  model: slot.modelName,
  version: slot.version,
  timeframe: slot.timeframe,
  runId: slot.runId,
  summary: slot.summary,
}));
const PRODUCT_LINE_1D_RUN_ID = PRODUCT_SLOTS.find((slot) => slot.id === "line-1d")?.runId ?? null;
const PRODUCT_BAND_1D_RUN_ID = PRODUCT_SLOTS.find((slot) => slot.id === "band-1d")?.runId ?? null;

const CONFIG_KEYS_COMMON = ["role", "model_role", "feature_set", "checkpoint_selection", "seq_len", "horizon", "wandb_status"];
const CONFIG_KEYS_LINE = ["patch_len", "stride", "patch_stride"];
const CONFIG_KEYS_BAND = ["q_low", "q_high", "lambda_band", "band_mode"];
const CONFIG_LABELS: Record<string, string> = {
  role: "역할",
  model_role: "모델 역할",
  feature_set: "사용 데이터",
  checkpoint_selection: "모델 선택 기준",
  seq_len: "입력 길이",
  horizon: "예측 기간",
  wandb_status: "실험 추적 상태",
  patch_len: "패치 길이",
  stride: "패치 간격",
  patch_stride: "패치 간격",
  q_low: "하단 분위수",
  q_high: "상단 분위수",
  lambda_band: "밴드 손실 가중치",
  band_mode: "밴드 방식",
};

const LINE_METRICS: MetricDefinition[] = [
  { label: "순위 상관", keys: ["ic_mean", "spearman_ic", "h1_h5_ic_mean", "all_horizon_ic_mean"] },
  { label: "상위-하위 수익 차", keys: ["long_short_spread", "h1_h5_long_short_spread", "all_horizon_long_short_spread"] },
  { label: "위험 오판율", keys: ["false_safe_tail_rate", "h1_h5_false_safe_tail_rate", "all_horizon_false_safe_tail_rate"], format: "rate" },
  { label: "큰 하락 포착률", keys: ["severe_downside_recall", "h1_h5_severe_downside_recall", "all_horizon_severe_downside_recall"], format: "rate" },
  { label: "수수료 반영 샤프", keys: ["fee_adjusted_sharpe", "h1_h5_fee_adjusted_sharpe", "all_horizon_fee_adjusted_sharpe"] },
];

const BAND_METRICS: MetricDefinition[] = [
  { label: "목표 포함률", keys: ["nominal_coverage", "h1_h5_band_nominal_coverage", "all_horizon_band_nominal_coverage"], format: "rate" },
  { label: "실제 포함률", keys: ["empirical_coverage", "h1_h5_band_empirical_coverage", "all_horizon_band_empirical_coverage"], format: "rate" },
  { label: "포함률 오차", keys: ["coverage_abs_error", "h1_h5_band_coverage_abs_error", "all_horizon_band_coverage_abs_error"], format: "pct_point" },
  { label: "하단 이탈률", keys: ["lower_breach_rate", "h1_h5_band_lower_breach_rate"], format: "rate" },
  { label: "상단 이탈률", keys: ["upper_breach_rate", "h1_h5_band_upper_breach_rate"], format: "rate" },
  { label: "평균 밴드 폭", keys: ["avg_band_width", "h1_h5_band_avg_band_width"] },
  { label: "비대칭 구간 점수", keys: ["asymmetric_interval_score", "h1_h5_band_asymmetric_interval_score", "all_horizon_band_asymmetric_interval_score"] },
  { label: "밴드 폭 반응도", keys: ["band_width_ic", "h1_h5_band_band_width_ic", "all_horizon_band_band_width_ic"] },
  { label: "하방 폭 반응도", keys: ["downside_width_ic", "h1_h5_band_downside_width_ic", "all_horizon_band_downside_width_ic"] },
];

const LINE_COMPARISON_METRICS: ComparisonMetricDefinition[] = [
  { id: "ic_mean", label: "순위 상관", keys: ["ic_mean", "spearman_ic", "h1_h5_ic_mean", "all_horizon_ic_mean"], better: "higher" },
  { id: "long_short_spread", label: "상위-하위 수익 차", keys: ["long_short_spread", "h1_h5_long_short_spread", "all_horizon_long_short_spread"], better: "higher" },
  { id: "fee_adjusted_sharpe", label: "수수료 반영 샤프", keys: ["fee_adjusted_sharpe", "h1_h5_fee_adjusted_sharpe", "all_horizon_fee_adjusted_sharpe"], better: "higher" },
  { id: "false_safe_tail_rate", label: "위험 오판율", keys: ["false_safe_tail_rate", "h1_h5_false_safe_tail_rate", "all_horizon_false_safe_tail_rate"], format: "rate", better: "lower" },
  { id: "false_safe_severe_rate", label: "큰 위험 오판율", keys: ["false_safe_severe_rate", "h1_h5_false_safe_severe_rate", "all_horizon_false_safe_severe_rate"], format: "rate", better: "lower" },
  { id: "severe_downside_recall", label: "큰 하락 포착률", keys: ["severe_downside_recall", "h1_h5_severe_downside_recall", "all_horizon_severe_downside_recall"], format: "rate", better: "higher" },
];

const BAND_COMPARISON_METRICS: ComparisonMetricDefinition[] = [
  { id: "empirical_coverage", label: "실제 포함률", keys: ["empirical_coverage", "h1_h5_band_empirical_coverage", "all_horizon_band_empirical_coverage"], format: "rate", better: "target_coverage" },
  { id: "coverage_abs_error", label: "포함률 오차", keys: ["coverage_abs_error", "h1_h5_band_coverage_abs_error", "all_horizon_band_coverage_abs_error"], format: "pct_point", better: "lower" },
  { id: "lower_breach_rate", label: "하단 이탈률", keys: ["lower_breach_rate", "h1_h5_band_lower_breach_rate"], format: "rate", better: "lower" },
  { id: "upper_breach_rate", label: "상단 이탈률", keys: ["upper_breach_rate", "h1_h5_band_upper_breach_rate"], format: "rate", better: "lower" },
  { id: "asymmetric_interval_score", label: "비대칭 구간 점수", keys: ["asymmetric_interval_score", "h1_h5_band_asymmetric_interval_score", "all_horizon_band_asymmetric_interval_score"], better: "lower" },
  { id: "avg_band_width", label: "평균 밴드 폭", keys: ["avg_band_width", "h1_h5_band_avg_band_width"], better: "neutral" },
  { id: "band_width_ic", label: "밴드 폭 반응도", keys: ["band_width_ic", "h1_h5_band_band_width_ic", "all_horizon_band_band_width_ic"], better: "higher" },
  { id: "downside_width_ic", label: "하방 폭 반응도", keys: ["downside_width_ic", "h1_h5_band_downside_width_ic", "all_horizon_band_downside_width_ic"], better: "higher" },
];

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

function formatKoreanDateTime(value: string | null | undefined) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return `${new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).format(date)} KST`;
}

function formatMetric(value: unknown, format: MetricDefinition["format"] = "number", fallback = "-") {
  if (value == null) {
    return fallback;
  }
  if (typeof value !== "number") {
    return formatValue(value);
  }
  if (!Number.isFinite(value)) {
    return fallback;
  }
  if (format === "rate") {
    return `${formatValue(value * 100, 1)}%`;
  }
  if (format === "pct_point") {
    return `${formatValue(value * 100, 1)}%p`;
  }
  return formatValue(value);
}

function formatStatusLabel(status: string | null | undefined) {
  if (status === "completed") {
    return "완료";
  }
  if (status === "failed_nan") {
    return "실패";
  }
  if (status === "failed_quality_gate") {
    return "기준 미달";
  }
  return status ?? "-";
}

function formatRoleLabel(role: string | null | undefined) {
  if (role === "line_model" || role === "line_v2" || role === "line") {
    return "보수적 기준선";
  }
  if (role === "band_model" || role === "band") {
    return "AI 밴드";
  }
  if (role === "composite_model") {
    return "이전 조합 실험";
  }
  return role ?? "-";
}

function formatModelLabel(value: unknown) {
  if (value === "patchtst") {
    return "PatchTST";
  }
  if (value === "cnn_lstm") {
    return "CNN-LSTM";
  }
  if (value === "line_band_composite") {
    return "결합 방식 실험";
  }
  return formatValue(value);
}

function formatFeatureSet(value: unknown) {
  if (value === "full_features") {
    return "전체 피처";
  }
  if (value === "price_volatility_volume") {
    return "가격·변동성·거래량";
  }
  return formatValue(value);
}

function formatConfigLabel(key: string) {
  return CONFIG_LABELS[key] ?? key;
}

function extractErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 설정과 백엔드 상태를 확인해주세요.";
    }
    return error.message;
  }
  return fallback;
}

function getMetricByKeys(metrics: Record<string, unknown> | null | undefined, keys: string[]) {
  if (!metrics) {
    return null;
  }
  for (const key of keys) {
    const value = metrics[key];
    if (value != null) {
      return value;
    }
  }
  return null;
}

function getMetricText(
  detail: AiRunDetail | null,
  definition: MetricDefinition,
  fallback: string,
  source: "test" | "val" = "test"
) {
  const metrics = source === "test" ? detail?.test_metrics : detail?.val_metrics;
  return formatMetric(getMetricByKeys(metrics, definition.keys), definition.format, fallback);
}

function getMetricNumber(detail: AiRunDetail | null, definition: MetricDefinition, source: "test" | "val" = "test") {
  const metrics = source === "test" ? detail?.test_metrics : detail?.val_metrics;
  const value = getMetricByKeys(metrics, definition.keys);
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function getMetricNumberFromStoredEvaluation(detail: AiRunDetail | null, definition: MetricDefinition) {
  return getMetricNumber(detail, definition, "test") ?? getMetricNumber(detail, definition, "val");
}

function hasStoredEvaluationMetrics(detail: AiRunDetail | null, definitions: MetricDefinition[]) {
  return definitions.some((definition) => getMetricNumberFromStoredEvaluation(detail, definition) != null);
}

function getProductMetricDefinitions(slot: ProductSlot) {
  if (slot.kind === "band") {
    return BAND_METRICS;
  }
  if (slot.kind === "line") {
    return LINE_METRICS;
  }
  return [];
}

function getProductSlotStatus(slot: ProductSlot, detail: AiRunDetail | null, isLoading: boolean): ProductSlotStatus {
  if (slot.kind === "preparing-line" || slot.kind === "preparing-band" || !slot.runId) {
    return "준비 중";
  }
  if (slot.runId) {
    return "사용 중";
  }
  if (detail && hasStoredEvaluationMetrics(detail, getProductMetricDefinitions(slot))) {
    return "사용 중";
  }
  return "연결 필요";
}

function getStatusPillClass(status: ProductSlotStatus) {
  if (status === "사용 중") {
    return "active";
  }
  if (status === "연결 필요") {
    return "warning";
  }
  return "pending";
}

function getComparisonDefinitions(kind: ExperimentKind) {
  return kind === "band" ? BAND_COMPARISON_METRICS : LINE_COMPARISON_METRICS;
}

function formatSignedNumber(value: number | null, digits = 4) {
  if (value == null) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatValue(value, digits)}`;
}

function formatSignedPctPoint(value: number | null) {
  if (value == null) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatValue(value, 1)}%p`;
}

function formatComparisonDiff(value: number, format: MetricDefinition["format"]) {
  if (format === "rate" || format === "pct_point") {
    return formatSignedPctPoint(value * 100);
  }
  return formatSignedNumber(value);
}

function getComparisonResult(metric: ComparisonMetricDefinition, productValue: number, experimentValue: number) {
  const tolerance = metric.format === "rate" || metric.format === "pct_point" ? 0.002 : 0.0005;
  if (metric.better === "neutral") {
    return "neutral" as const;
  }
  if (metric.better === "target_coverage") {
    const productDistance = Math.abs(productValue - 0.7);
    const experimentDistance = Math.abs(experimentValue - 0.7);
    if (Math.abs(productDistance - experimentDistance) <= tolerance) {
      return "similar" as const;
    }
    return experimentDistance < productDistance ? "better" as const : "worse" as const;
  }
  const diff = experimentValue - productValue;
  if (Math.abs(diff) <= tolerance) {
    return "similar" as const;
  }
  if (metric.better === "higher") {
    return diff > 0 ? "better" as const : "worse" as const;
  }
  return diff < 0 ? "better" as const : "worse" as const;
}

function getMetricInterpretation(metric: ComparisonMetricDefinition, result: ComparisonRow["result"], productValue: number, experimentValue: number) {
  if (metric.id === "ic_mean") {
    return result === "better" ? "방향/순위 구분은 현재 모델보다 좋았습니다." : result === "worse" ? "방향/순위 구분력이 현재 모델보다 약합니다." : "방향/순위 구분력은 현재 모델과 비슷합니다.";
  }
  if (metric.id === "long_short_spread") {
    return result === "better" ? "좋은 종목과 나쁜 종목을 나누는 힘은 더 좋았습니다." : result === "worse" ? "좋은 종목과 나쁜 종목을 나누는 힘이 약합니다." : "상위/하위 구분력은 비슷합니다.";
  }
  if (metric.id === "fee_adjusted_sharpe") {
    return result === "better" ? "수수료 반영 안정성은 더 좋았습니다." : result === "worse" ? "수수료 반영 안정성이 현재 모델보다 약합니다." : "수수료 반영 안정성은 비슷합니다.";
  }
  if (metric.id === "false_safe_tail_rate" || metric.id === "false_safe_severe_rate") {
    return result === "better" ? "위험 구간을 안전하다고 본 비율은 더 낮았습니다." : result === "worse" ? "위험 구간을 안전하다고 본 비율이 높습니다." : "위험 오판율은 비슷합니다.";
  }
  if (metric.id === "severe_downside_recall") {
    return result === "better" ? "큰 하락을 포착하는 힘은 더 좋았습니다." : result === "worse" ? "큰 하락을 포착하는 힘이 약합니다." : "큰 하락 포착력은 비슷합니다.";
  }
  if (metric.id === "empirical_coverage") {
    return result === "better" ? "목표 포함률에 더 가까웠습니다." : result === "worse" ? "목표 포함률과 실제 포함률의 차이가 더 큽니다." : "목표 포함률과의 거리는 비슷합니다.";
  }
  if (metric.id === "coverage_abs_error") {
    return result === "better" ? "포함률 오차는 더 작았습니다." : result === "worse" ? "목표 포함률과 실제 포함률 차이가 더 큽니다." : "포함률 오차는 비슷합니다.";
  }
  if (metric.id === "lower_breach_rate") {
    return result === "better" ? "하단 이탈률은 더 낮았습니다." : result === "worse" ? "하방 위험을 충분히 덮지 못했습니다." : "하방 위험 커버는 비슷합니다.";
  }
  if (metric.id === "upper_breach_rate") {
    return result === "better" ? "상단 이탈률은 더 낮았습니다." : result === "worse" ? "상방 변동 범위가 부족했습니다." : "상방 변동 커버는 비슷합니다.";
  }
  if (metric.id === "asymmetric_interval_score") {
    return result === "better" ? "하방 페널티를 포함한 종합 점수는 더 좋았습니다." : result === "worse" ? "하방 페널티를 포함한 종합 품질이 약합니다." : "비대칭 구간 점수는 비슷합니다.";
  }
  if (metric.id === "avg_band_width") {
    if (experimentValue > productValue) {
      return "밴드가 더 넓어 보수적이지만 화면 해석은 무거워질 수 있습니다.";
    }
    if (experimentValue < productValue) {
      return "밴드가 더 좁아 보이지만 위험을 덜 덮을 수 있습니다.";
    }
    return "밴드 폭은 비슷합니다.";
  }
  if (metric.id === "band_width_ic") {
    return result === "better" ? "변동성이 커질 때 밴드가 더 잘 넓어졌습니다." : result === "worse" ? "변동성이 커질 때 밴드가 같이 넓어지는 반응이 약합니다." : "변동성 반응은 비슷합니다.";
  }
  if (metric.id === "downside_width_ic") {
    return result === "better" ? "하락 위험이 커질 때 밴드 반응은 더 좋았습니다." : result === "worse" ? "하락 위험이 커질 때 밴드가 반응하는 힘이 약합니다." : "하방 위험 반응은 비슷합니다.";
  }
  return result === "better" ? "제품 모델보다 나은 지표입니다." : result === "worse" ? "제품 모델보다 약한 지표입니다." : "제품 모델과 비슷합니다.";
}

function buildComparisonRows(detail: AiRunDetail, productDetail: AiRunDetail | null) {
  const kind = getExperimentKind(detail);
  if (!kind || !productDetail) {
    return [];
  }
  return getComparisonDefinitions(kind)
    .map((metric): ComparisonRow | null => {
      const productValue = getMetricNumber(productDetail, metric);
      const experimentValue = getMetricNumber(detail, metric);
      if (productValue == null || experimentValue == null) {
        return null;
      }
      const result = getComparisonResult(metric, productValue, experimentValue);
      return {
        id: metric.id,
        label: metric.label,
        productValue,
        experimentValue,
        productText: formatMetric(productValue, metric.format),
        experimentText: formatMetric(experimentValue, metric.format),
        diffText: formatComparisonDiff(experimentValue - productValue, metric.format),
        interpretation: getMetricInterpretation(metric, result, productValue, experimentValue),
        result,
      };
    })
    .filter((row): row is ComparisonRow => row != null);
}

function hasDisplayableComparison(detail: AiRunDetail, productDetail: AiRunDetail | null) {
  if (detail.timeframe === "1W") {
    return hasDisplayableExperimentMetrics(detail);
  }
  return buildComparisonRows(detail, productDetail).length >= 2;
}

function getConfigValue(detail: AiRunDetail | null, key: string) {
  if (!detail) {
    return null;
  }
  if (key === "horizon") {
    return detail.horizon ?? detail.config_summary?.horizon ?? null;
  }
  return detail.config_summary?.[key] ?? (detail as unknown as Record<string, unknown>)[key] ?? null;
}

function normalizeRunRole(role: unknown): string | null {
  const normalized = String(role ?? "").trim().toLowerCase();
  if (normalized === "line_model" || normalized === "line_v2" || normalized === "line") {
    return "line_model";
  }
  if (normalized === "band_model" || normalized === "band") {
    return "band_model";
  }
  if (normalized === "composite_model" || normalized === "composite") {
    return "composite_model";
  }
  return null;
}

function getRunRole(run: AiRunSummary | AiRunDetail | null): string | null {
  if (!run) {
    return null;
  }
  if (run.run_id === PRODUCT_LINE_1D_RUN_ID) {
    return "line_model";
  }
  if (run.run_id === PRODUCT_BAND_1D_RUN_ID) {
    return "band_model";
  }
  if ("config_summary" in run) {
    const configRole = normalizeRunRole(run.config_summary?.role) ?? normalizeRunRole(run.config_summary?.model_role);
    if (configRole) {
      return configRole;
    }
  }
  return normalizeRunRole(run.role);
}

function isLegacyRun(run: AiRunSummary | AiRunDetail) {
  const modelName = String(run.model_name ?? "");
  const role = getRunRole(run);
  return run.is_legacy || modelName === "line_band_composite" || role === "composite_model";
}

function getExperimentKind(run: AiRunSummary | AiRunDetail): ExperimentKind | null {
  const role = getRunRole(run);
  if (role === "line_model") {
    return "line";
  }
  if (role === "band_model") {
    return "band";
  }
  return null;
}

function getConfigKeys(detail: AiRunDetail | null) {
  const role = getRunRole(detail);
  if (role === "line_model") {
    return [...CONFIG_KEYS_COMMON, ...CONFIG_KEYS_LINE];
  }
  if (role === "band_model") {
    return [...CONFIG_KEYS_COMMON, ...CONFIG_KEYS_BAND];
  }
  return [...CONFIG_KEYS_COMMON, ...CONFIG_KEYS_LINE, ...CONFIG_KEYS_BAND];
}

function formatConfigValue(key: string, value: unknown) {
  if (key === "role" || key === "model_role") {
    return formatRoleLabel(typeof value === "string" ? value : null);
  }
  if (key === "feature_set") {
    return formatFeatureSet(value);
  }
  return formatValue(value);
}

function formatExperimentName(run: AiRunSummary | AiRunDetail) {
  const modelName = String(run.model_name ?? "");
  const horizon = run.horizon ?? ("config_summary" in run ? getConfigValue(run, "horizon") : null);
  const horizonLabel = horizon ? `h${formatValue(horizon, 0)}` : "h?";
  if (modelName === "patchtst") {
    const patchLen = "config_summary" in run ? getConfigValue(run, "patch_len") : null;
    const stride = "config_summary" in run ? getConfigValue(run, "stride") ?? getConfigValue(run, "patch_stride") : null;
    const seqLen = "config_summary" in run ? getConfigValue(run, "seq_len") : null;
    const epochs = "config_summary" in run ? getConfigValue(run, "epochs") : null;
    const featureSet = "config_summary" in run ? getConfigValue(run, "feature_set") ?? run.feature_set : run.feature_set;
    const checkpointSelection = "config_summary" in run ? getConfigValue(run, "checkpoint_selection") ?? run.checkpoint_selection : run.checkpoint_selection;
    if (horizon === 20) {
      return "PatchTST h20 예측선 실험";
    }
    if (featureSet && featureSet !== "full_features") {
      return `PatchTST ${horizonLabel} no fundamentals 예측선`;
    }
    if (typeof patchLen === "number" && patchLen >= 32) {
      return `PatchTST ${horizonLabel} 긴 패치 예측선`;
    }
    if (typeof stride === "number" && stride <= 4) {
      return `PatchTST ${horizonLabel} Dense 예측선`;
    }
    if (typeof seqLen === "number" && seqLen <= 60 && typeof epochs === "number" && epochs >= 30) {
      return `PatchTST ${horizonLabel} seq60 장기 학습 예측선`;
    }
    if (typeof seqLen === "number" && seqLen <= 60) {
      return `PatchTST ${horizonLabel} seq60 초기 예측선`;
    }
    if (checkpointSelection === "line_gate") {
      return `PatchTST ${horizonLabel} Line Gate 예측선`;
    }
    if (checkpointSelection === "val_total") {
      return `PatchTST ${horizonLabel} Val Total 예측선`;
    }
    return `PatchTST ${horizonLabel} 기본 예측선`;
  }
  if (modelName === "cnn_lstm") {
    const qLow = "config_summary" in run ? getConfigValue(run, "q_low") : null;
    const qHigh = "config_summary" in run ? getConfigValue(run, "q_high") : null;
    const featureSet = "config_summary" in run ? getConfigValue(run, "feature_set") ?? run.feature_set : run.feature_set;
    if (typeof qLow === "number" && typeof qHigh === "number") {
      const dataLabel = featureSet === "price_volatility_volume" ? "가격·변동성" : "";
      return `CNN-LSTM q${Math.round(qLow * 100)} ${dataLabel} AI 밴드`.replace(/\s+/g, " ").trim();
    }
    if (featureSet === "price_volatility_volume") {
      return "CNN-LSTM 가격·변동성·거래량 밴드";
    }
    return "CNN-LSTM 밴드 초기 실험";
  }
  return `${formatModelLabel(run.model_name)} 실험`;
}

function getExperimentDescription(detail: AiRunDetail | null, category: ExperimentCategory) {
  if (category === "quality_failed") {
    return "목표 기준에 미치지 못해 현재 제품 화면에는 쓰지 않는 실험입니다.";
  }
  return "현재 제품 화면에는 쓰지 않지만, 모델 구조와 실험 방향을 비교하기 위해 남겨둔 실행입니다.";
}

function getExperimentTag(run: AiRunSummary | AiRunDetail, category: ExperimentCategory) {
  if (category === "quality_failed") {
    return "기준 미달";
  }
  if (run.status !== "completed") {
    return "보류";
  }
  if (run.created_at && run.created_at < "2026-01-01") {
    return "개선 전 버전";
  }
  return "제품 미사용";
}

function getChangedExperimentFields(detail: AiRunDetail) {
  const fields: string[] = [];
  const experimentKind = getExperimentKind(detail);
  const featureSet = detail.feature_set ?? getConfigValue(detail, "feature_set");
  const horizon = detail.horizon ?? getConfigValue(detail, "horizon");
  const checkpointSelection = detail.checkpoint_selection ?? getConfigValue(detail, "checkpoint_selection");

  if (horizon != null) {
    fields.push(`예측 기간 h${formatValue(horizon, 0)}`);
  }
  if (featureSet) {
    fields.push(`사용 데이터 ${formatFeatureSet(featureSet)}`);
  }
  if (checkpointSelection) {
    fields.push(`모델 선택 기준 ${formatValue(checkpointSelection)}`);
  }
  if (experimentKind === "line") {
    const patchLen = getConfigValue(detail, "patch_len");
    const stride = getConfigValue(detail, "stride") ?? getConfigValue(detail, "patch_stride");
    const seqLen = getConfigValue(detail, "seq_len");
    if (patchLen != null) {
      fields.push(`패치 길이 ${formatValue(patchLen, 0)}`);
    }
    if (stride != null) {
      fields.push(`패치 간격 ${formatValue(stride, 0)}`);
    }
    if (seqLen != null) {
      fields.push(`입력 길이 ${formatValue(seqLen, 0)}`);
    }
  }
  if (experimentKind === "band") {
    const qLow = getConfigValue(detail, "q_low");
    const qHigh = getConfigValue(detail, "q_high");
    const bandMode = getConfigValue(detail, "band_mode");
    if (qLow != null && qHigh != null) {
      fields.push(`분위수 ${formatValue(qLow)} / ${formatValue(qHigh)}`);
    }
    if (bandMode) {
      fields.push(`밴드 방식 ${formatValue(bandMode)}`);
    }
  }
  return fields.length > 0 ? fields : ["실험 조건 상세는 상세 정보에서 확인"];
}

function GoalCard({ title, target, actual, diff, judgement, description, tone = "neutral" }: GoalCardProps) {
  return (
    <article className={`goal-card goal-card--${tone}`}>
      <div className="goal-card__topline">
        <strong>{title}</strong>
        <span>{judgement}</span>
      </div>
      <div className="goal-card__rows">
        <div>
          <span>목표</span>
          <strong>{target}</strong>
        </div>
        <div>
          <span>실제</span>
          <strong>{actual}</strong>
        </div>
        <div>
          <span>차이</span>
          <strong>{diff}</strong>
        </div>
      </div>
      <p>{description}</p>
    </article>
  );
}

function GoalCardGrid({ cards }: { cards: GoalCardData[] }) {
  return (
    <div className="goal-grid">
      {cards.map((card) => (
        <GoalCard
          key={card.id}
          title={card.title}
          target={card.target}
          actual={card.actual}
          diff={card.diff}
          judgement={card.judgement}
          description={card.description}
          tone={card.tone}
        />
      ))}
    </div>
  );
}

function DataList({ items }: { items: string[] }) {
  return (
    <ul className="model-data-list">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

const DETAIL_LABELS: Record<string, string> = {
  ...CONFIG_LABELS,
  model_name: "모델 구조",
  model_ver: "모델 버전",
  status: "상태",
  timeframe: "차트 단위",
  feature_version: "피처 버전",
  line_target_type: "예측선 목표값",
  band_target_type: "밴드 목표값",
  learning_rate: "학습률",
  lr: "학습률",
  weight_decay: "가중치 감쇠",
  batch_size: "배치 크기",
  epochs: "학습 epoch",
  best_epoch: "선택 epoch",
  best_val_total: "검증 손실",
  device: "학습 장치",
  amp_dtype: "혼합 정밀도",
  seed: "시드",
  n_features: "피처 수",
  hidden_size: "은닉 크기",
  kernel_size: "커널 크기",
  fp32_modules: "FP32 모듈",
  n_heads: "어텐션 헤드",
  n_layers: "레이어 수",
  d_model: "모델 차원",
  alpha: "alpha",
  beta: "beta",
  lambda_line: "예측선 손실 가중치",
  ci_aggregate: "CI 집계 기준",
  ci_target_fast: "빠른 CI 목표",
  run_id: "실행 ID",
  created_at: "생성 시각",
  checkpoint_exists: "체크포인트",
  wandb_run_id: "실험 추적 ID",
};

const DETAIL_HIDDEN_KEYS = new Set([
  "checkpoint_path",
  "line_model_run_id",
  "band_model_run_id",
  "composition_policy",
  "band_calibration_method",
  "band_calibration_params",
  "prediction_composition_version",
  "deprecated_for_phase1_product_contract",
  "indicator_layer_replacement",
  "line_model_name",
  "band_model_name",
]);

const DETAIL_GROUPS = [
  {
    id: "training",
    title: "학습 설정",
    keys: ["epochs", "batch_size", "lr", "learning_rate", "weight_decay", "dropout", "device", "amp_dtype", "seed", "best_epoch", "best_val_total"],
  },
  {
    id: "data",
    title: "데이터 설정",
    keys: ["timeframe", "horizon", "seq_len", "feature_set", "feature_version", "n_features", "target", "line_target_type", "band_target_type", "ci_aggregate", "ci_target_fast"],
  },
  {
    id: "structure",
    title: "모델 구조",
    keys: ["model_name", "model_ver", "patch_len", "patch_stride", "stride", "d_model", "n_heads", "n_layers", "hidden_size", "kernel_size", "fp32_modules", "band_mode"],
  },
  {
    id: "loss",
    title: "손실/평가 설정",
    keys: ["alpha", "beta", "lambda_line", "lambda_band", "q_low", "q_high", "checkpoint_selection"],
  },
];

function isDetailObject(value: unknown) {
  return typeof value === "object" && value !== null;
}

function shouldShowDetailValue(key: string, value: unknown) {
  if (DETAIL_HIDDEN_KEYS.has(key) || value == null) {
    return false;
  }
  if (typeof value === "number") {
    return Number.isFinite(value);
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (typeof value === "boolean") {
    return true;
  }
  if (Array.isArray(value)) {
    return value.length > 0 && value.every((item) => !isDetailObject(item));
  }
  return false;
}

function formatDetailLabel(key: string) {
  return DETAIL_LABELS[key] ?? key.replace(/_/g, " ");
}

function formatDetailValue(key: string, value: unknown) {
  if (key === "model_name") {
    return formatModelLabel(value);
  }
  if (key === "role") {
    return formatRoleLabel(typeof value === "string" ? value : null);
  }
  if (key === "status") {
    return formatStatusLabel(typeof value === "string" ? value : null);
  }
  if (key === "feature_set") {
    return formatFeatureSet(value);
  }
  if (typeof value === "boolean") {
    return value ? "사용" : "미사용";
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatValue(item)).join(", ");
  }
  return formatValue(value);
}

function buildDetailValueMap(detail: AiRunDetail) {
  const values: Record<string, unknown> = {
    ...detail.config_summary,
    model_name: detail.model_name,
    model_ver: detail.model_ver,
    status: detail.status,
    role: getRunRole(detail),
    timeframe: detail.timeframe,
    horizon: detail.horizon,
    feature_set: detail.feature_set ?? detail.config_summary?.feature_set,
    feature_version: detail.feature_version,
    line_target_type: detail.line_target_type,
    band_target_type: detail.band_target_type,
    checkpoint_selection: detail.checkpoint_selection ?? detail.config_summary?.checkpoint_selection,
    wandb_status: detail.wandb_status ?? detail.config_summary?.wandb_status,
    best_epoch: detail.best_epoch,
    best_val_total: detail.best_val_total,
  };
  return values;
}

function buildDetailFields(detail: AiRunDetail, keys: string[], usedKeys: Set<string>) {
  const values = buildDetailValueMap(detail);
  return keys.flatMap((key) => {
    const value = values[key];
    if (!shouldShowDetailValue(key, value) || usedKeys.has(key)) {
      return [];
    }
    usedKeys.add(key);
    return [{
      key,
      label: formatDetailLabel(key),
      value: formatDetailValue(key, value),
    }];
  });
}

function buildAdditionalFields(detail: AiRunDetail, usedKeys: Set<string>) {
  const values = buildDetailValueMap(detail);
  return Object.keys(values)
    .sort()
    .flatMap((key) => {
      const value = values[key];
      if (!shouldShowDetailValue(key, value) || usedKeys.has(key)) {
        return [];
      }
      usedKeys.add(key);
      return [{
        key,
        label: formatDetailLabel(key),
        value: formatDetailValue(key, value),
      }];
    });
}

function getPredictionDescription(detail: AiRunDetail) {
  const horizon = detail.horizon ? `${detail.horizon}거래일` : "다음 구간";
  const kind = getExperimentKind(detail);
  if (kind === "band") {
    return `${formatModelLabel(detail.model_name)} 모델이 ${horizon}의 예상 변동 범위를 계산합니다. 밴드는 매수 목표가 아니라 위험 범위를 이해하기 위한 보조 지표입니다.`;
  }
  return `${formatModelLabel(detail.model_name)} 모델이 ${horizon}의 수익 방향과 종목 순위 판단을 돕는 예측선을 계산합니다. 단독 매매 신호가 아니라 가격 차트 위의 참고선으로 사용합니다.`;
}

function getStructureDescription(detail: AiRunDetail) {
  const modelName = String(detail.model_name ?? "");
  if (modelName === "patchtst") {
    return "PatchTST는 시계열을 패치 단위로 나누어 최근 가격·지표 흐름의 패턴을 학습하는 Transformer 계열 구조입니다.";
  }
  if (modelName === "cnn_lstm") {
    return "CNN-LSTM은 합성곱으로 짧은 구간의 패턴을 잡고, LSTM으로 시간 순서의 흐름을 이어서 보는 구조입니다.";
  }
  if (modelName === "tide") {
    return "TiDE는 과거 구간을 인코딩하고 미래 구간을 디코딩하는 시계열 예측 구조입니다.";
  }
  return "이 실행은 저장된 설정과 평가 지표를 기준으로 구조와 품질을 확인합니다.";
}

function getMetricTargetLabel(metric: MetricDefinition) {
  const key = metric.keys[0];
  if (["ic_mean", "spearman_ic", "long_short_spread", "fee_adjusted_sharpe", "band_width_ic", "downside_width_ic"].includes(key)) {
    return "0보다 큼";
  }
  if (key === "false_safe_tail_rate") {
    return "25% 이하";
  }
  if (key === "severe_downside_recall") {
    return "70% 이상";
  }
  if (key === "nominal_coverage" || key === "empirical_coverage") {
    return "70% 근처";
  }
  if (key === "coverage_abs_error") {
    return "5%p 이하";
  }
  if (key === "lower_breach_rate" || key === "upper_breach_rate") {
    return "15% 근처";
  }
  if (key === "asymmetric_interval_score") {
    return "낮을수록 좋음";
  }
  if (key === "avg_band_width") {
    return "관찰 지표";
  }
  return "해석 기준 확인";
}

function DetailFieldGrid({ fields }: { fields: DetailField[] }) {
  if (fields.length === 0) {
    return <div className="compact-note">표시할 값이 없습니다.</div>;
  }
  return (
    <div className="detail-field-grid">
      {fields.map((field) => (
        <div key={field.key} className="detail-field">
          <span>{field.label}</span>
          <strong className={field.monospace ? "detail-field__mono" : undefined}>{field.value}</strong>
        </div>
      ))}
    </div>
  );
}

function ModelRunDetails({ detail, metricDefinitions }: { detail: AiRunDetail | null; metricDefinitions: MetricDefinition[] }) {
  if (!detail) {
    return null;
  }
  const usedKeys = new Set<string>();
  const groupedFields = DETAIL_GROUPS.map((group) => ({
    ...group,
    fields: buildDetailFields(detail, group.keys, usedKeys),
  })).filter((group) => group.fields.length > 0);
  const additionalFields = buildAdditionalFields(detail, usedKeys);
  const metricRows = metricDefinitions.map((metric) => ({
    ...metric,
    target: getMetricTargetLabel(metric),
    testValue: getMetricText(detail, metric, "-", "test"),
    valValue: getMetricText(detail, metric, "-", "val"),
  })).filter((metric) => metric.testValue !== "-" || metric.valValue !== "-");
  const wandbStatusValue = detail.wandb_status ?? getConfigValue(detail, "wandb_status");
  const storageFields: DetailField[] = [
    { key: "run_id", label: "실행 ID", value: detail.run_id, monospace: true },
    { key: "status", label: "상태", value: formatStatusLabel(detail.status) },
    { key: "checkpoint_exists", label: "체크포인트", value: detail.checkpoint_path ? "저장됨" : "없음" },
    ...(wandbStatusValue != null ? [{ key: "wandb_status", label: "실험 추적 상태", value: formatDetailValue("wandb_status", wandbStatusValue) }] : []),
    ...(detail.wandb_run_id ? [{ key: "wandb_run_id", label: "실험 추적 ID", value: detail.wandb_run_id, monospace: true }] : []),
    ...(detail.created_at ? [{ key: "created_at", label: "생성 시각", value: formatKoreanDateTime(detail.created_at) }] : []),
  ].filter((field) => shouldShowDetailValue(field.key, field.value));

  return (
    <details className="model-run-details">
      <summary className="model-run-details__summary">
        <span>상세 정보</span>
        <em>모델 설정·평가 지표·저장 정보</em>
      </summary>
      <div className="model-run-details__content">
        <div className="model-run-details__header">
          <div>
            <span className="eyebrow">모델 설정</span>
            <h3>{formatModelLabel(detail.model_name)} · {formatRoleLabel(getRunRole(detail))}</h3>
            <p>{getPredictionDescription(detail)}</p>
          </div>
          <div className="detail-status-card">
            <span>버전</span>
            <strong>{detail.model_ver ?? "v1"}</strong>
            <em>{formatStatusLabel(detail.status)}</em>
          </div>
        </div>

        <div className="model-detail-section">
          <h4>모델 구조</h4>
          <p>{getStructureDescription(detail)}</p>
        </div>

        {groupedFields.map((group) => (
          <div key={group.id} className="model-detail-section">
            <h4>{group.title}</h4>
            <DetailFieldGrid fields={group.fields} />
          </div>
        ))}

        {additionalFields.length > 0 ? (
          <div className="model-detail-section">
            <h4>추가 설정</h4>
            <DetailFieldGrid fields={additionalFields} />
          </div>
        ) : null}

        {metricRows.length > 0 ? (
          <div className="model-detail-section">
            <h4>평가 지표</h4>
            <div className="detail-metric-grid">
              {metricRows.map((metric) => (
                <div key={metric.label}>
                  <span>{metric.label}</span>
                  <strong>목표 {metric.target}</strong>
                  <em>test {metric.testValue}</em>
                  <em>val {metric.valValue}</em>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="model-detail-section model-detail-section--storage">
          <h4>저장 정보</h4>
          <DetailFieldGrid fields={storageFields} />
        </div>
      </div>
    </details>
  );
}

function ProductSlotCard({
  slot,
  status,
  active,
  onSelect,
}: {
  slot: ProductSlot;
  status: ProductSlotStatus;
  active: boolean;
  onSelect: (slotId: ProductSlotId) => void;
}) {
  return (
    <button
      type="button"
      className={`product-slot-card${active ? " product-slot-card--active" : ""}`}
      onClick={() => onSelect(slot.id)}
    >
      <div className="product-slot-card__header">
        <span className={`status-pill status-pill--${getStatusPillClass(status)}`}>{status}</span>
        <span>{slot.timeframe}</span>
      </div>
      <strong>{slot.title}</strong>
      <p>{slot.summary}</p>
      <div className="product-slot-card__meta">
        <span>{slot.version ? `${slot.model} · ${slot.version}` : slot.model}</span>
      </div>
    </button>
  );
}

function ExperimentButton({
  run,
  detail,
  active,
  category,
  onSelect,
}: {
  run: AiRunSummary;
  detail: AiRunDetail;
  active: boolean;
  category: ExperimentCategory;
  onSelect: (runId: string, category: ExperimentCategory) => void;
}) {
  return (
    <button
      type="button"
      className={`experiment-row${active ? " experiment-row--active" : ""}`}
      onClick={() => onSelect(run.run_id, category)}
    >
      <strong>{formatExperimentName(detail)}</strong>
      <span>
        {getExperimentKind(detail) === "band" ? "밴드 실험" : "예측선 실험"} · {detail.timeframe ?? "-"} · h{detail.horizon ?? "-"}
      </span>
      <em>{getExperimentTag(run, category)}</em>
    </button>
  );
}

function ExperimentDisclosure({
  title,
  items,
  selected,
  onSelect,
}: {
  title: string;
  items: ExperimentListItem[];
  selected: SelectedItem;
  onSelect: (runId: string, category: ExperimentCategory) => void;
}) {
  return (
    <details className="experiment-disclosure">
      <summary>
        <span>{title}</span>
        <em>{items.length}개</em>
      </summary>
      {items.length > 0 ? (
        <div className="experiment-row-list">
          {items.map((item) => (
            <ExperimentButton
              key={`${item.category}-${item.run.run_id}`}
              run={item.run}
              detail={item.detail}
              category={item.category}
              active={selected.kind === "experiment" && selected.runId === item.run.run_id}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : (
        <div className="compact-note">표시할 실험이 없습니다.</div>
      )}
    </details>
  );
}

function PreparingSlotDetail({ slot }: { slot: ProductSlot }) {
  const description =
    slot.id === "line-1w"
      ? "1W 보수적 기준선은 v1에서 제공하지 않습니다. 1W AI 밴드는 자동 갱신 상태로 별도 카드에서 확인할 수 있습니다."
      : "이 슬롯은 다음 학습 단계에서 채울 예정입니다.";

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className="status-pill status-pill--pending">준비 중</span>
        <h2>{slot.title}</h2>
        <p>{description}</p>
      </div>
      <div className="goal-grid">
        <GoalCard
          title="제품 모델"
          judgement="준비 중"
          target="검증 완료"
          actual="아직 없음"
          diff="검증 필요"
          description="현재는 주간 AI 밴드만 활성 상태입니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다."
          tone="neutral"
        />
      </div>
    </div>
  );
}

function StoredEvaluationSection({ cards }: { cards: GoalCardData[] }) {
  return (
    <section>
      <div className="panel-heading panel-heading--compact">
        <h3>목표 대비 평가</h3>
      </div>
      <div className="trust-note">이 값은 저장된 평가 결과 기준입니다. 평가가 없으면 성능을 판단하지 않습니다.</div>
      {cards.length > 0 ? <GoalCardGrid cards={cards} /> : <div className="empty-state empty-state--compact">저장된 평가 없음</div>}
    </section>
  );
}

function LineModelDetail({ detail, slot }: { detail: AiRunDetail | null; slot?: ProductSlot | null }) {
  const cards = detail ? buildLineExperimentCards(detail) : [];
  const hasEvaluation = cards.length > 0;
  const isWeekly = (detail?.timeframe ?? slot?.timeframe) === "1W";
  const status: ProductSlotStatus = slot?.runId ? "사용 중" : "준비 중";
  const horizonLabel = isWeekly ? "4주" : "5거래일";
  const title = slot?.title ?? (isWeekly ? "1W 보수적 기준선" : "1D 보수적 기준선");
  const summary =
    slot?.summary ??
    (isWeekly
      ? "1W 보수적 기준선은 v1에서 제공하지 않습니다."
      : "수익 방향과 종목 순위 판단에는 사용할 수 있지만, 위험 회피 품질은 개선 중입니다.");

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className={`status-pill status-pill--${getStatusPillClass(status)}`}>{status}</span>
        <h2>{title}</h2>
        <p>{summary}</p>
      </div>

      <section className="model-story-grid">
        <article>
          <h3>모델 역할</h3>
          <p>
            Line v2 기반 제품 모델입니다. 최근 데이터를 보고 앞으로 {horizonLabel} 후 도착가를 보수적으로 추정합니다.
            출력은 수익률 단위 score이며 화면에서는 기준 종가를 곱해 가격으로 환산합니다.
          </p>
        </article>
        <article>
          <h3>사용 데이터</h3>
          <DataList items={["가격", "거래량", "기술적 지표", "재무·거시·시장 폭 관련 피처"]} />
        </article>
      </section>

      <div className="notice">
        계약: 수익률 단위 score. asof 종가 × (1 + score)로 가격 환산 후 표시합니다.
        출력 의미: 5거래일 후 도착가 보수 추정 (h5 horizon, β=5).
      </div>

      {hasEvaluation ? (
        <>
          <div className="notice">현재 품질 판정: 저장된 평가 결과 기준으로 방향성과 순위 판단은 확인됐지만, 위험 오판율 개선이 필요합니다.</div>
          <section className="model-story-grid">
            <article>
              <h3>좋은 점</h3>
              <ul className="model-copy-list">
                <li>저장된 평가 지표로 방향성과 순위 판단 신호를 확인했습니다.</li>
                <li>상위-하위 구분력은 목표 대비 평가 카드에서 확인할 수 있습니다.</li>
                <li>수수료 반영 안정성도 저장된 metric 기준으로만 판단합니다.</li>
              </ul>
            </article>
            <article>
              <h3>아쉬운 점</h3>
              <ul className="model-copy-list">
                <li>위험 오판율과 큰 하락 포착률은 계속 개선해야 합니다.</li>
                <li>단독 매매 신호라기보다는 참고선으로 보는 것이 맞습니다.</li>
              </ul>
            </article>
          </section>
        </>
      ) : (
        <div className="notice">저장된 평가 없음: 평가 metric이 확인되기 전에는 성능을 판단하지 않습니다.</div>
      )}

      <StoredEvaluationSection cards={cards} />

      {hasEvaluation ? <div className="notice">다음 개선 방향: 위험 오판율을 낮추는 loss/selector 개선이 다음 과제입니다.</div> : null}
      <div className="notice">이 모델은 투자 조언이 아니라 보조 판단선입니다. 특히 위험 회피 품질은 계속 개선 중입니다.</div>
      <ModelRunDetails detail={detail} metricDefinitions={LINE_METRICS} />
    </div>
  );
}

function BandModelDetail({ detail, slot }: { detail: AiRunDetail | null; slot?: ProductSlot | null }) {
  const cards = detail ? buildBandExperimentCards(detail) : [];
  const hasEvaluation = cards.length > 0;
  const status: ProductSlotStatus = slot?.runId ? "사용 중" : "준비 중";
  const isWeekly = (detail?.timeframe ?? slot?.timeframe) === "1W";
  const horizonLabel = isWeekly ? "4주" : "5거래일";
  const title = slot?.title ?? (isWeekly ? "1W AI 밴드 v1" : "1D AI 밴드 v1");
  const summary =
    slot?.summary ??
    (isWeekly
      ? "1W 예상 변동 범위를 보여주는 주간 리스크 참고 밴드입니다."
      : "저장된 평가 결과가 확인된 1D 위험 범위 보조지표입니다.");

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className={`status-pill status-pill--${getStatusPillClass(status)}`}>{status}</span>
        <h2>{title}</h2>
        <p>{summary}</p>
      </div>

      <section className="model-story-grid">
        <article>
          <h3>모델 역할</h3>
          <p>
            TiDE 기반 제품 밴드입니다. 최근 가격·변동성 흐름을 보고 앞으로 {horizonLabel}의 예상 변동 범위를 계산합니다.
            밴드가 넓어지는 구간은 모델이 더 큰 변동 가능성을 보는 구간입니다.
          </p>
        </article>
        <article>
          <h3>사용 데이터</h3>
          <DataList items={["가격", "변동성", "거래량"]} />
        </article>
      </section>

      {hasEvaluation ? (
        <>
          <div className="notice">현재 품질 판정: 저장된 평가 결과 기준으로 목표 포함률과 밴드 반응도를 확인했습니다.</div>
          <section className="model-story-grid">
            <article>
              <h3>좋은 점</h3>
              <ul className="model-copy-list">
                <li>저장된 평가 metric으로 포함률과 이탈률을 확인했습니다.</li>
                <li>밴드 폭 반응도는 목표 대비 평가 카드에서 확인할 수 있습니다.</li>
                <li>밴드는 매수/매도 신호가 아니라 위험 범위로 해석합니다.</li>
              </ul>
            </article>
            <article>
              <h3>아쉬운 점</h3>
              <ul className="model-copy-list">
                <li>종목별로 과하게 넓거나 좁게 보일 수 있어 계속 검증이 필요합니다.</li>
                <li>밴드가 넓다고 수익 기회라는 뜻은 아닙니다.</li>
              </ul>
            </article>
          </section>
        </>
      ) : (
        <div className="notice">저장된 평가 없음: 평가 metric이 확인되기 전에는 성능을 판단하지 않습니다.</div>
      )}

      <StoredEvaluationSection cards={cards} />

      {hasEvaluation ? <div className="notice">다음 개선 방향: 종목별 밴드 폭 안정성과 하방 반응도 개선이 다음 과제입니다.</div> : null}
      <div className="notice">AI 밴드는 수익 목표가 아니라 위험 범위입니다. 가격 목표선처럼 해석하지 마세요.</div>
      <ModelRunDetails detail={detail} metricDefinitions={BAND_METRICS} />
    </div>
  );
}

function buildLineExperimentCards(detail: AiRunDetail): GoalCardData[] {
  const cards: GoalCardData[] = [];
  const ic = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[0]);
  const spread = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[1]);
  const falseSafe = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[2]);
  const recall = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[3]);

  if (ic != null) {
    cards.push({
      id: "ic",
      title: "순위 상관",
      judgement: ic > 0 ? "통과" : "개선 필요",
      target: "0보다 큼",
      actual: formatMetric(ic),
      diff: formatSignedNumber(ic),
      description: ic > 0 ? "수익 방향을 어느 정도 구분했습니다." : "순위 상관이 약해 수익 방향을 안정적으로 구분하지 못했습니다.",
      tone: ic > 0 ? "good" : "warn",
    });
  }
  if (spread != null) {
    cards.push({
      id: "spread",
      title: "상위-하위 수익 차",
      judgement: spread > 0 ? "통과" : "개선 필요",
      target: "0보다 큼",
      actual: formatMetric(spread),
      diff: formatSignedNumber(spread),
      description: spread > 0 ? "높게 본 구간이 낮게 본 구간보다 나았습니다." : "상위 구간과 하위 구간의 성과 차이가 약했습니다.",
      tone: spread > 0 ? "good" : "warn",
    });
  }
  if (falseSafe != null) {
    cards.push({
      id: "false-safe",
      title: "위험 오판율",
      judgement: falseSafe <= 0.25 ? "통과" : "개선 필요",
      target: "25% 이하",
      actual: formatMetric(falseSafe, "rate"),
      diff: formatSignedPctPoint((falseSafe - 0.25) * 100),
      description: falseSafe <= 0.25 ? "위험 구간 오판이 목표 안에 있습니다." : "위험 구간을 안전하다고 보는 경우가 많았습니다.",
      tone: falseSafe <= 0.25 ? "good" : "warn",
    });
  }
  if (recall != null) {
    cards.push({
      id: "downside-recall",
      title: "큰 하락 포착률",
      judgement: recall >= 0.7 ? "통과" : "개선 필요",
      target: "70% 이상",
      actual: formatMetric(recall, "rate"),
      diff: formatSignedPctPoint((recall - 0.7) * 100),
      description: recall >= 0.7 ? "큰 하락 구간을 비교적 잘 포착했습니다." : "큰 하락을 모두 잡아내기에는 아직 부족합니다.",
      tone: recall >= 0.7 ? "good" : "warn",
    });
  }
  return cards;
}

function buildBandExperimentCards(detail: AiRunDetail): GoalCardData[] {
  const cards: GoalCardData[] = [];
  const empirical = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[1]);
  const coverageError = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[2]);
  const lower = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[3]);
  const upper = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[4]);
  const widthIc = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[7]);

  if (empirical != null) {
    cards.push({
      id: "coverage",
      title: "실제 포함률",
      judgement: Math.abs(empirical - 0.7) <= 0.05 ? "통과" : "개선 필요",
      target: "70%",
      actual: formatMetric(empirical, "rate"),
      diff: formatSignedPctPoint((empirical - 0.7) * 100),
      description: empirical >= 0.65 ? "목표 포함률에 비교적 가깝습니다." : "실제 포함률이 목표보다 낮아 위험 범위를 충분히 덮지 못했습니다.",
      tone: Math.abs(empirical - 0.7) <= 0.05 ? "good" : "warn",
    });
  }
  if (coverageError != null) {
    cards.push({
      id: "coverage-error",
      title: "포함률 오차",
      judgement: coverageError <= 0.05 ? "통과" : "개선 필요",
      target: "5%p 이하",
      actual: formatMetric(coverageError, "pct_point"),
      diff: coverageError <= 0.05 ? "기준 안" : formatSignedPctPoint((coverageError - 0.05) * 100),
      description: coverageError <= 0.05 ? "목표 포함률과의 차이가 허용 범위에 있습니다." : "목표 포함률과 실제 포함률의 차이가 큽니다.",
      tone: coverageError <= 0.05 ? "good" : "warn",
    });
  }
  if (lower != null) {
    cards.push({
      id: "lower",
      title: "하단 이탈률",
      judgement: lower <= 0.15 ? "통과" : "개선 필요",
      target: "15% 근처",
      actual: formatMetric(lower, "rate"),
      diff: formatSignedPctPoint((lower - 0.15) * 100),
      description: lower <= 0.15 ? "하방 위험을 어느 정도 덮었습니다." : "하단 이탈률이 높아 하방 위험을 충분히 덮지 못했습니다.",
      tone: lower <= 0.15 ? "good" : "warn",
    });
  }
  if (upper != null) {
    cards.push({
      id: "upper",
      title: "상단 이탈률",
      judgement: upper <= 0.15 ? "통과" : "보통",
      target: "15% 근처",
      actual: formatMetric(upper, "rate"),
      diff: formatSignedPctPoint((upper - 0.15) * 100),
      description: upper <= 0.15 ? "상단 방향도 비교적 안정적으로 덮었습니다." : "상단 방향 이탈이 다소 많습니다.",
      tone: upper <= 0.15 ? "good" : "neutral",
    });
  }
  if (widthIc != null) {
    cards.push({
      id: "width-ic",
      title: "밴드 폭 반응도",
      judgement: widthIc > 0 ? "통과" : "개선 필요",
      target: "0보다 큼",
      actual: formatMetric(widthIc),
      diff: formatSignedNumber(widthIc),
      description: widthIc > 0 ? "실제 변동성이 큰 구간에서 밴드가 넓어지는 경향이 있습니다." : "변동성이 커지는 구간에 밴드가 충분히 반응하지 못했습니다.",
      tone: widthIc > 0 ? "good" : "warn",
    });
  }
  return cards;
}

function buildExperimentCards(detail: AiRunDetail) {
  return getExperimentKind(detail) === "band" ? buildBandExperimentCards(detail) : buildLineExperimentCards(detail);
}

function hasDisplayableExperimentMetrics(detail: AiRunDetail) {
  return buildExperimentCards(detail).length >= 2;
}

function getExperimentFailureReason(detail: AiRunDetail) {
  const kind = getExperimentKind(detail);
  if (kind === "band") {
    const empirical = getMetricNumber(detail, BAND_METRICS[1]);
    const lower = getMetricNumber(detail, BAND_METRICS[3]);
    const widthIc = getMetricNumber(detail, BAND_METRICS[7]);
    if (empirical != null && empirical < 0.65) {
      return "실제 포함률이 목표보다 낮아 위험 범위를 충분히 덮지 못했습니다.";
    }
    if (lower != null && lower > 0.15) {
      return "하단 이탈률이 높아 하방 위험을 충분히 덮지 못했습니다.";
    }
    if (widthIc != null && widthIc <= 0) {
      return "변동성이 커지는 구간에 밴드 폭이 충분히 반응하지 못했습니다.";
    }
    return "현재 제품 모델보다 설명력이나 안정성에서 우선순위가 낮아 제품 화면에는 쓰지 않습니다.";
  }

  const ic = getMetricNumber(detail, LINE_METRICS[0]);
  const falseSafe = getMetricNumber(detail, LINE_METRICS[2]);
  const recall = getMetricNumber(detail, LINE_METRICS[3]);
  if (ic != null && ic <= 0) {
    return "순위 상관이 약해 수익 방향을 안정적으로 구분하지 못했습니다.";
  }
  if (falseSafe != null && falseSafe > 0.25) {
    return "위험 오판율이 높아 위험 구간을 안전하다고 보는 경우가 많았습니다.";
  }
  if (recall != null && recall < 0.7) {
    return "큰 하락을 모두 잡아내기에는 아직 부족했습니다.";
  }
  return "현재 제품 모델보다 품질이나 해석 우선순위가 낮아 제품 화면에는 쓰지 않습니다.";
}

function describeWeaknesses(rows: ComparisonRow[]) {
  const weakRows = rows.filter((row) => row.result === "worse");
  if (weakRows.length === 0) {
    return "제품 모델보다 뚜렷하게 약한 비교 지표는 제한적입니다. 다만 현재 제품 모델이 더 안정적인 기준으로 쓰이고 있어 이 실험은 이전 실험으로 남겼습니다.";
  }
  return weakRows
    .slice(0, 3)
    .map((row) => `${row.label}: ${row.interpretation}`)
    .join(" ");
}

function describeStrengths(rows: ComparisonRow[]) {
  const betterRows = rows.filter((row) => row.result === "better");
  if (betterRows.length === 0) {
    return "제품 모델보다 더 좋았던 핵심 비교 지표는 확인되지 않았습니다.";
  }
  return betterRows
    .slice(0, 3)
    .map((row) => `${row.label}은 더 좋았습니다.`)
    .join(" ");
}

function getComparisonVerdictTag(detail: AiRunDetail, rows: ComparisonRow[], category: ExperimentCategory) {
  if (detail.timeframe === "1W") {
    return "제품 기준 미확정";
  }
  if (category === "quality_failed") {
    return "제품 후보 탈락";
  }
  if (rows.some((row) => row.result === "worse")) {
    return "이전 실험";
  }
  return "보류";
}

function getFinalJudgement(detail: AiRunDetail, rows: ComparisonRow[], category: ExperimentCategory) {
  if (detail.timeframe === "1W") {
    return "1W 보수적 기준선은 v1에서 제공하지 않지만 1W AI 밴드는 활성 (CP178 walk-forward lower calibration)입니다. 이 실험 결과를 현재 1W 제품 모델 대비 우열로 과장하지 않습니다.";
  }
  const kind = getExperimentKind(detail);
  const weakRows = rows.filter((row) => row.result === "worse");
  const betterRows = rows.filter((row) => row.result === "better");
  const roleText = kind === "band" ? "AI 밴드" : "보수적 기준선";
  if (weakRows.length > 0 && betterRows.length > 0) {
    return `${formatExperimentName(detail)}은 ${betterRows[0].label}에서는 제품 모델보다 나은 면이 있었지만, ${weakRows[0].label}에서 약해 ${roleText} 제품 모델로 쓰기 어렵습니다.`;
  }
  if (weakRows.length > 0) {
    return `${formatExperimentName(detail)}은 ${weakRows[0].label} 지표가 현재 제품 모델보다 약해 ${roleText} 제품 모델로 선택하지 않았습니다.`;
  }
  if (category === "quality_failed") {
    return "품질 기준을 통과하지 못해 제품 화면에는 쓰지 않습니다.";
  }
  return "일부 지표는 제품 모델과 비슷했지만, 현재 제품 모델을 대체할 만큼 명확한 우위가 확인되지 않아 이전 실험으로 남겼습니다.";
}

function ComparisonTable({ rows }: { rows: ComparisonRow[] }) {
  if (rows.length === 0) {
    return <div className="compact-note">제품 기준 미확정 상태입니다.</div>;
  }
  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>항목</th>
            <th>제품 모델</th>
            <th>이 실험</th>
            <th>차이</th>
            <th>해석</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.label}</td>
              <td>{row.productText}</td>
              <td>{row.experimentText}</td>
              <td>{row.diffText}</td>
              <td>{row.interpretation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExperimentDetail({
  detail,
  category,
  productDetail,
}: {
  detail: AiRunDetail | null;
  category: ExperimentCategory;
  productDetail: AiRunDetail | null;
}) {
  if (!detail) {
    return <div className="empty-state">실험을 선택하면 상세 설명을 표시합니다.</div>;
  }
  const role = getRunRole(detail);
  const metrics = role === "band_model" ? BAND_METRICS : LINE_METRICS;
  const experimentKind = getExperimentKind(detail);
  const changedFields = getChangedExperimentFields(detail);
  const comparisonRows = buildComparisonRows(detail, productDetail);
  const verdict = getComparisonVerdictTag(detail, comparisonRows, category);

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className="status-pill status-pill--muted">{verdict}</span>
        <h2>{formatExperimentName(detail)}</h2>
        <p>{getExperimentDescription(detail, category)}</p>
      </div>

      <section className="model-story-grid">
        <article>
          <h3>역할</h3>
          <p>{experimentKind === "band" ? "AI 밴드 실험입니다. 예상 변동 범위가 목표 비율에 맞게 실제 수익률을 덮는지 확인합니다." : "예측선 실험입니다. 수익 방향과 위험 구간을 얼마나 안정적으로 구분하는지 확인합니다."}</p>
        </article>
        <article>
          <h3>실험에서 바꾼 것</h3>
          <DataList items={changedFields} />
        </article>
      </section>

      <section className="model-story-grid">
        <article>
          <h3>제품 모델 대비 부족했던 점</h3>
          <p>{detail.timeframe === "1W" ? "1W 제품 기준이 아직 확정되지 않아 1D 제품 모델처럼 직접 비교하지 않습니다." : describeWeaknesses(comparisonRows)}</p>
        </article>
        <article>
          <h3>제품 모델 대비 좋았던 점</h3>
          <p>{detail.timeframe === "1W" ? "주간 제품 기준 확정 뒤 다시 평가할 수 있습니다." : describeStrengths(comparisonRows)}</p>
        </article>
      </section>

      <section>
        <div className="panel-heading panel-heading--compact">
          <h3>비교 지표</h3>
        </div>
        <ComparisonTable rows={comparisonRows} />
      </section>

      <section className="model-story-grid">
        <article>
          <h3>최종 판단</h3>
          <p>{getFinalJudgement(detail, comparisonRows, category)}</p>
        </article>
        <article>
          <h3>다음 확인 방향</h3>
          <p>{detail.timeframe === "1W" ? "1W 제품 기준이 확정된 뒤 같은 지표로 다시 비교합니다." : "실험 조건 상세는 상세 정보에서 확인하고, 제품 모델보다 나았던 지표를 다음 학습 조건에 반영합니다."}</p>
        </article>
      </section>
      <ModelRunDetails detail={detail} metricDefinitions={metrics} />
    </div>
  );
}

export default function TrainingView() {
  const [runs, setRuns] = useState<AiRunSummary[]>([]);
  const [failedQualityRuns, setFailedQualityRuns] = useState<AiRunSummary[]>([]);
  const [experimentDetails, setExperimentDetails] = useState<Record<string, AiRunDetail>>({});
  const [productLineDetail, setProductLineDetail] = useState<AiRunDetail | null>(null);
  const [productBandDetail, setProductBandDetail] = useState<AiRunDetail | null>(null);
  const [productWeeklyLineDetail, setProductWeeklyLineDetail] = useState<AiRunDetail | null>(null);
  const [selected, setSelected] = useState<SelectedItem>({ kind: "slot", slotId: "line-1d" });
  const [detail, setDetail] = useState<AiRunDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const experimentGroups = useMemo(() => {
    const candidates = [
        ...runs
          .filter((run) => !PRODUCT_RUN_IDS.has(run.run_id))
          .filter((run) => !isLegacyRun(run))
          .map((run) => ({ run, category: "previous" as const })),
        ...failedQualityRuns
          .filter((run) => !isLegacyRun(run))
          .map((run) => ({ run, category: "quality_failed" as const })),
      ];
    const displayable = candidates
      .map((item) => {
        const itemDetail = experimentDetails[item.run.run_id];
        const kind = itemDetail ? getExperimentKind(itemDetail) : null;
        const productDetail = kind === "band" ? productBandDetail : productLineDetail;
        if (!itemDetail || !kind || !hasDisplayableComparison(itemDetail, productDetail)) {
          return null;
        }
        return {
          ...item,
          detail: itemDetail,
          kind,
          tag: getExperimentTag(item.run, item.category),
        };
      })
      .filter((item): item is ExperimentListItem => item != null);
    const seenExperimentNames = new Set<string>();
    const uniqueDisplayable = displayable.filter((item) => {
      const key = `${item.kind}-${item.category}-${formatExperimentName(item.detail)}`;
      if (seenExperimentNames.has(key)) {
        return false;
      }
      seenExperimentNames.add(key);
      return true;
    });
    return {
      line: uniqueDisplayable.filter((item) => item.kind === "line"),
      band: uniqueDisplayable.filter((item) => item.kind === "band"),
    };
  }, [runs, failedQualityRuns, experimentDetails, productLineDetail, productBandDetail]);

  async function loadDetail(selection: SelectedItem, runId: string | null) {
    setSelected(selection);
    setErrorMessage(null);
    if (selection.kind === "slot") {
      setDetail(null);
      return;
    }
    if (!runId) {
      setDetail(null);
      return;
    }
    setIsDetailLoading(true);
    try {
      const detailResponse = await fetchAiRun(runId, { includeConfig: false });
      setDetail(detailResponse.data);
    } catch (error) {
      setDetail(null);
      setErrorMessage(extractErrorMessage(error, "실행 상세를 불러오지 못했습니다."));
    } finally {
      setIsDetailLoading(false);
    }
  }

  async function loadRuns() {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const [completedResult, failedQualityResult] = await Promise.allSettled([
        fetchAiRuns({ status: "completed", modelName: "", includeLegacy: true, limit: 100 }),
        fetchAiRuns({ status: "failed_quality_gate", modelName: "", includeLegacy: true, limit: 100 }),
      ]);
      const completedRuns = completedResult.status === "fulfilled" ? completedResult.value.data : [];
      const qualityRuns = failedQualityResult.status === "fulfilled" ? failedQualityResult.value.data : [];
      const filterModelRuns = (items: AiRunSummary[]) =>
        items.filter((run) => Boolean(getRunRole(run)) || TRAINING_RUN_MODELS.has(String(run.model_name ?? "")));
      const filteredCompletedRuns = filterModelRuns(completedRuns);
      const filteredQualityRuns = filterModelRuns(qualityRuns);
      const [productLineResult, productBandResult] = await Promise.allSettled([
        PRODUCT_LINE_1D_RUN_ID ? fetchAiRun(PRODUCT_LINE_1D_RUN_ID, { includeConfig: false }) : Promise.resolve(null),
        PRODUCT_BAND_1D_RUN_ID ? fetchAiRun(PRODUCT_BAND_1D_RUN_ID, { includeConfig: false }) : Promise.resolve(null),
      ]);
      const nextProductLineDetail = productLineResult.status === "fulfilled" ? productLineResult.value?.data ?? null : null;
      const nextProductBandDetail = productBandResult.status === "fulfilled" ? productBandResult.value?.data ?? null : null;
      const nextProductWeeklyLineDetail = null;
      const experimentCandidates = [...filteredCompletedRuns, ...filteredQualityRuns]
        .filter((run) => !PRODUCT_RUN_IDS.has(run.run_id))
        .filter((run) => !isLegacyRun(run));
      const detailResults = await Promise.allSettled(
        experimentCandidates.map(async (run) => {
          const response = await fetchAiRun(run.run_id, { includeConfig: false });
          return [run.run_id, response.data] as const;
        })
      );
      const nextExperimentDetails: Record<string, AiRunDetail> = {};
      detailResults.forEach((result) => {
        if (result.status === "fulfilled") {
          const [runId, runDetail] = result.value;
          if (hasDisplayableExperimentMetrics(runDetail)) {
            nextExperimentDetails[runId] = runDetail;
          }
        }
      });

      setRuns(filteredCompletedRuns);
      setFailedQualityRuns(filteredQualityRuns);
      setProductLineDetail(nextProductLineDetail);
      setProductBandDetail(nextProductBandDetail);
      setProductWeeklyLineDetail(nextProductWeeklyLineDetail);
      setExperimentDetails(nextExperimentDetails);
      if (PRODUCT_LINE_1D_RUN_ID) {
        await loadDetail({ kind: "slot", slotId: "line-1d" }, PRODUCT_LINE_1D_RUN_ID);
      }
    } catch (error) {
      setRuns([]);
      setFailedQualityRuns([]);
      setExperimentDetails({});
      setProductLineDetail(null);
      setProductBandDetail(null);
      setProductWeeklyLineDetail(null);
      setDetail(null);
      setErrorMessage(extractErrorMessage(error, "AI 모델 목록을 불러오지 못했습니다."));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadRuns();
  }, []);

  const selectedSlot = selected.kind === "slot" ? PRODUCT_SLOTS.find((slot) => slot.id === selected.slotId) ?? PRODUCT_SLOTS[0] : null;
  const selectedExperimentCategory = selected.kind === "experiment" ? selected.category : "previous";
  const selectedExperimentProductDetail = detail && getExperimentKind(detail) === "band" ? productBandDetail : productLineDetail;
  const getLoadedProductDetailForSlot = (slot: ProductSlot) => {
    if (slot.id === "line-1d") {
      return productLineDetail;
    }
    if (slot.id === "line-1w") {
      return productWeeklyLineDetail;
    }
    if (slot.id === "band-1d") {
      return productBandDetail;
    }
    return null;
  };

  return (
    <div className="view-stack">
      <header className="view-header">
        <div className="view-header__title">
          <div className="eyebrow">제품 모델 설명</div>
          <h1>AI 모델</h1>
          <p>Lens는 예측선 모델과 AI 밴드 모델을 분리해서 평가하고, 검증된 모델만 주식 보기 화면에 사용합니다.</p>
        </div>
      </header>

      {errorMessage ? <div className="notice notice--error">{errorMessage}</div> : null}

      <section className="panel model-status-panel">
        <div className="panel-heading">
          <div className="eyebrow">제품 모델 현황</div>
          <h2>현재 사용 상태</h2>
        </div>
        <div className="product-slot-grid">
          {PRODUCT_SLOTS.map((slot) => (
            <ProductSlotCard
              key={slot.id}
              slot={slot}
              status={getProductSlotStatus(slot, getLoadedProductDetailForSlot(slot), isLoading)}
              active={selected.kind === "slot" && selected.slotId === slot.id}
              onSelect={(slotId) => {
                const nextSlot = PRODUCT_SLOTS.find((item) => item.id === slotId);
                void loadDetail({ kind: "slot", slotId }, nextSlot?.runId ?? null);
              }}
            />
          ))}
        </div>
      </section>

      <section className="panel model-detail-panel">
        {isDetailLoading || isLoading ? (
          <div className="empty-state">AI 모델 정보를 불러오는 중입니다.</div>
        ) : selectedSlot?.kind === "line" ? (
          <LineModelDetail detail={detail} slot={selectedSlot} />
        ) : selectedSlot?.kind === "band" ? (
          <BandModelDetail detail={detail} slot={selectedSlot} />
        ) : selectedSlot ? (
          <PreparingSlotDetail slot={selectedSlot} />
        ) : (
          <ExperimentDetail detail={detail} category={selectedExperimentCategory} productDetail={selectedExperimentProductDetail} />
        )}
      </section>

      <section className="panel model-experiment-panel">
        <div className="panel-heading">
          <h2>이전 실험</h2>
        </div>
        <div className="experiment-disclosure-grid">
          <ExperimentDisclosure
            title="예측선 실험 보기"
            items={experimentGroups.line}
            selected={selected}
            onSelect={(runId, nextCategory) => void loadDetail({ kind: "experiment", runId, category: nextCategory }, runId)}
          />
          <ExperimentDisclosure
            title="밴드 실험 보기"
            items={experimentGroups.band}
            selected={selected}
            onSelect={(runId, nextCategory) => void loadDetail({ kind: "experiment", runId, category: nextCategory }, runId)}
          />
        </div>
      </section>
    </div>
  );
}
