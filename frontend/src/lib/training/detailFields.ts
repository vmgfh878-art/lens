// AI run 의 상세 정보 (config, metrics) 를 사용자에게 보여주기 위한 라벨링 / 그룹화.

import type { AiRunDetail } from "@/api/client";

import { CONFIG_LABELS, MetricDefinition } from "./constants";
import {
  formatFeatureSet,
  formatModelLabel,
  formatRoleLabel,
  formatStatusLabel,
  formatValue,
} from "./formatters";
import { getExperimentKind, getRunRole } from "./runUtils";

export interface DetailField {
  key: string;
  label: string;
  value: string;
  monospace?: boolean;
}

export const DETAIL_LABELS: Record<string, string> = {
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

export const DETAIL_HIDDEN_KEYS = new Set([
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

export const DETAIL_GROUPS = [
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

export function shouldShowDetailValue(key: string, value: unknown) {
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

export function formatDetailLabel(key: string) {
  return DETAIL_LABELS[key] ?? key.replace(/_/g, " ");
}

export function formatDetailValue(key: string, value: unknown) {
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

export function buildDetailValueMap(detail: AiRunDetail) {
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

export function buildDetailFields(detail: AiRunDetail, keys: string[], usedKeys: Set<string>): DetailField[] {
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

export function buildAdditionalFields(detail: AiRunDetail, usedKeys: Set<string>): DetailField[] {
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

export function getPredictionDescription(detail: AiRunDetail) {
  const horizon = detail.horizon ? `${detail.horizon}거래일` : "다음 구간";
  const kind = getExperimentKind(detail);
  if (kind === "band") {
    return `${formatModelLabel(detail.model_name)} 모델이 ${horizon}의 예상 변동 범위를 계산합니다. 밴드는 매수 목표가 아니라 위험 범위를 이해하기 위한 보조 지표입니다.`;
  }
  return `${formatModelLabel(detail.model_name)} 모델이 ${horizon}의 수익 방향과 종목 순위 판단을 돕는 예측선을 계산합니다. 단독 매매 신호가 아니라 가격 차트 위의 참고선으로 사용합니다.`;
}

export function getStructureDescription(detail: AiRunDetail) {
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

export function getMetricTargetLabel(metric: MetricDefinition) {
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
