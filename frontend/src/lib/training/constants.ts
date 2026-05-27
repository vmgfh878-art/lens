// AI 모델 페이지의 metric / config / product slot 정의.
// 데이터 정의이지 화면 로직 아님 → component 와 분리한다.

import type { ProductSlotId } from "@/lib/productSlots";
import { PRODUCT_SLOTS as PRODUCT_SLOT_CONFIGS } from "@/lib/productSlots";

export type ExperimentKind = "line" | "band";
export type ExperimentCategory = "previous" | "quality_failed";
export type ProductSlotKind = "line" | "band" | "preparing-line" | "preparing-band";
export type ProductSlotStatus = "사용 중" | "데이터 확인 중" | "연결 필요" | "준비 중";

export interface ProductSlot {
  id: ProductSlotId;
  kind: ProductSlotKind;
  title: string;
  model: string;
  version: string | null;
  timeframe: "1D" | "1W";
  runId: string | null;
  summary: string;
}

export interface MetricDefinition {
  label: string;
  keys: string[];
  format?: "number" | "rate" | "pct_point";
}

export interface ComparisonMetricDefinition extends MetricDefinition {
  id: string;
  better: "higher" | "lower" | "target_coverage" | "neutral";
}

export const TRAINING_RUN_MODELS = new Set([
  "patchtst",
  "cnn_lstm",
  "tide",
  "line_band_composite",
]);

export const PRODUCT_SLOTS: ProductSlot[] = PRODUCT_SLOT_CONFIGS.map((slot) => ({
  id: slot.id,
  kind:
    slot.status === "deferred"
      ? slot.kind === "line"
        ? "preparing-line"
        : "preparing-band"
      : slot.kind,
  title: slot.title,
  model: slot.modelName,
  version: slot.version,
  timeframe: slot.timeframe,
  runId: slot.runId,
  summary: slot.summary,
}));

export const PRODUCT_LINE_1D_RUN_ID =
  PRODUCT_SLOTS.find((slot) => slot.id === "line-1d")?.runId ?? null;
export const PRODUCT_BAND_1D_RUN_ID =
  PRODUCT_SLOTS.find((slot) => slot.id === "band-1d")?.runId ?? null;

export const CONFIG_KEYS_COMMON = [
  "role",
  "model_role",
  "feature_set",
  "checkpoint_selection",
  "seq_len",
  "horizon",
  "wandb_status",
];
export const CONFIG_KEYS_LINE = ["patch_len", "stride", "patch_stride"];
export const CONFIG_KEYS_BAND = ["q_low", "q_high", "lambda_band", "band_mode"];
export const CONFIG_LABELS: Record<string, string> = {
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

export const LINE_METRICS: MetricDefinition[] = [
  { label: "순위 상관", keys: ["ic_mean", "spearman_ic", "h1_h5_ic_mean", "all_horizon_ic_mean"] },
  { label: "상위-하위 수익 차", keys: ["long_short_spread", "h1_h5_long_short_spread", "all_horizon_long_short_spread"] },
  { label: "위험 오판율", keys: ["false_safe_tail_rate", "h1_h5_false_safe_tail_rate", "all_horizon_false_safe_tail_rate"], format: "rate" },
  { label: "큰 하락 포착률", keys: ["severe_downside_recall", "h1_h5_severe_downside_recall", "all_horizon_severe_downside_recall"], format: "rate" },
  { label: "수수료 반영 샤프", keys: ["fee_adjusted_sharpe", "h1_h5_fee_adjusted_sharpe", "all_horizon_fee_adjusted_sharpe"] },
];

export const BAND_METRICS: MetricDefinition[] = [
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

export const LINE_COMPARISON_METRICS: ComparisonMetricDefinition[] = [
  { id: "ic_mean", label: "순위 상관", keys: ["ic_mean", "spearman_ic", "h1_h5_ic_mean", "all_horizon_ic_mean"], better: "higher" },
  { id: "long_short_spread", label: "상위-하위 수익 차", keys: ["long_short_spread", "h1_h5_long_short_spread", "all_horizon_long_short_spread"], better: "higher" },
  { id: "fee_adjusted_sharpe", label: "수수료 반영 샤프", keys: ["fee_adjusted_sharpe", "h1_h5_fee_adjusted_sharpe", "all_horizon_fee_adjusted_sharpe"], better: "higher" },
  { id: "false_safe_tail_rate", label: "위험 오판율", keys: ["false_safe_tail_rate", "h1_h5_false_safe_tail_rate", "all_horizon_false_safe_tail_rate"], format: "rate", better: "lower" },
  { id: "false_safe_severe_rate", label: "큰 위험 오판율", keys: ["false_safe_severe_rate", "h1_h5_false_safe_severe_rate", "all_horizon_false_safe_severe_rate"], format: "rate", better: "lower" },
  { id: "severe_downside_recall", label: "큰 하락 포착률", keys: ["severe_downside_recall", "h1_h5_severe_downside_recall", "all_horizon_severe_downside_recall"], format: "rate", better: "higher" },
];

export const BAND_COMPARISON_METRICS: ComparisonMetricDefinition[] = [
  { id: "empirical_coverage", label: "실제 포함률", keys: ["empirical_coverage", "h1_h5_band_empirical_coverage", "all_horizon_band_empirical_coverage"], format: "rate", better: "target_coverage" },
  { id: "coverage_abs_error", label: "포함률 오차", keys: ["coverage_abs_error", "h1_h5_band_coverage_abs_error", "all_horizon_band_coverage_abs_error"], format: "pct_point", better: "lower" },
  { id: "lower_breach_rate", label: "하단 이탈률", keys: ["lower_breach_rate", "h1_h5_band_lower_breach_rate"], format: "rate", better: "lower" },
  { id: "upper_breach_rate", label: "상단 이탈률", keys: ["upper_breach_rate", "h1_h5_band_upper_breach_rate"], format: "rate", better: "lower" },
  { id: "asymmetric_interval_score", label: "비대칭 구간 점수", keys: ["asymmetric_interval_score", "h1_h5_band_asymmetric_interval_score", "all_horizon_band_asymmetric_interval_score"], better: "lower" },
  { id: "avg_band_width", label: "평균 밴드 폭", keys: ["avg_band_width", "h1_h5_band_avg_band_width"], better: "neutral" },
  { id: "band_width_ic", label: "밴드 폭 반응도", keys: ["band_width_ic", "h1_h5_band_band_width_ic", "all_horizon_band_band_width_ic"], better: "higher" },
  { id: "downside_width_ic", label: "하방 폭 반응도", keys: ["downside_width_ic", "h1_h5_band_downside_width_ic", "all_horizon_band_downside_width_ic"], better: "higher" },
];
