// AI run (line/band/composite) 역할 식별 / 분류 / 명명.

import type { AiRunDetail, AiRunSummary } from "@/api/client";

import {
  CONFIG_KEYS_BAND,
  CONFIG_KEYS_COMMON,
  CONFIG_KEYS_LINE,
  ExperimentCategory,
  ExperimentKind,
  PRODUCT_BAND_1D_RUN_ID,
  PRODUCT_LINE_1D_RUN_ID,
} from "./constants";
import {
  formatFeatureSet,
  formatModelLabel,
  formatRoleLabel,
  formatValue,
} from "./formatters";

export function getConfigValue(detail: AiRunDetail | null, key: string) {
  if (!detail) {
    return null;
  }
  if (key === "horizon") {
    return detail.horizon ?? detail.config_summary?.horizon ?? null;
  }
  return detail.config_summary?.[key] ?? (detail as unknown as Record<string, unknown>)[key] ?? null;
}

export function normalizeRunRole(role: unknown): string | null {
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

export function getRunRole(run: AiRunSummary | AiRunDetail | null): string | null {
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
    const configRole =
      normalizeRunRole(run.config_summary?.role) ?? normalizeRunRole(run.config_summary?.model_role);
    if (configRole) {
      return configRole;
    }
  }
  return normalizeRunRole(run.role);
}

export function isLegacyRun(run: AiRunSummary | AiRunDetail) {
  const modelName = String(run.model_name ?? "");
  const role = getRunRole(run);
  return run.is_legacy || modelName === "line_band_composite" || role === "composite_model";
}

export function getExperimentKind(run: AiRunSummary | AiRunDetail): ExperimentKind | null {
  const role = getRunRole(run);
  if (role === "line_model") {
    return "line";
  }
  if (role === "band_model") {
    return "band";
  }
  return null;
}

export function getConfigKeys(detail: AiRunDetail | null) {
  const role = getRunRole(detail);
  if (role === "line_model") {
    return [...CONFIG_KEYS_COMMON, ...CONFIG_KEYS_LINE];
  }
  if (role === "band_model") {
    return [...CONFIG_KEYS_COMMON, ...CONFIG_KEYS_BAND];
  }
  return [...CONFIG_KEYS_COMMON, ...CONFIG_KEYS_LINE, ...CONFIG_KEYS_BAND];
}

export function formatConfigValue(key: string, value: unknown) {
  if (key === "role" || key === "model_role") {
    return formatRoleLabel(typeof value === "string" ? value : null);
  }
  if (key === "feature_set") {
    return formatFeatureSet(value);
  }
  return formatValue(value);
}

export function formatExperimentName(run: AiRunSummary | AiRunDetail) {
  const modelName = String(run.model_name ?? "");
  const horizon = run.horizon ?? ("config_summary" in run ? getConfigValue(run, "horizon") : null);
  const horizonLabel = horizon ? `h${formatValue(horizon, 0)}` : "h?";
  if (modelName === "patchtst") {
    const patchLen = "config_summary" in run ? getConfigValue(run, "patch_len") : null;
    const stride =
      "config_summary" in run ? getConfigValue(run, "stride") ?? getConfigValue(run, "patch_stride") : null;
    const seqLen = "config_summary" in run ? getConfigValue(run, "seq_len") : null;
    const epochs = "config_summary" in run ? getConfigValue(run, "epochs") : null;
    const featureSet =
      "config_summary" in run ? getConfigValue(run, "feature_set") ?? run.feature_set : run.feature_set;
    const checkpointSelection =
      "config_summary" in run
        ? getConfigValue(run, "checkpoint_selection") ?? run.checkpoint_selection
        : run.checkpoint_selection;
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
    const featureSet =
      "config_summary" in run ? getConfigValue(run, "feature_set") ?? run.feature_set : run.feature_set;
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

export function getExperimentDescription(_detail: AiRunDetail | null, category: ExperimentCategory) {
  if (category === "quality_failed") {
    return "목표 기준에 미치지 못해 현재 제품 화면에는 쓰지 않는 실험입니다.";
  }
  return "현재 제품 화면에는 쓰지 않지만, 모델 구조와 실험 방향을 비교하기 위해 남겨둔 실행입니다.";
}

export function getExperimentTag(run: AiRunSummary | AiRunDetail, category: ExperimentCategory) {
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

export function getChangedExperimentFields(detail: AiRunDetail) {
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
