// AI 모델 페이지에서 값 표시용 helper.
// 일반 lib/formatters 와 별도로, training 도메인 특화 라벨 / 단위 매핑을 담는다.

import { CONFIG_LABELS, MetricDefinition } from "./constants";

export function formatValue(value: unknown, digits = 4) {
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

export function formatKoreanDateTime(value: string | null | undefined) {
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

export function formatMetric(
  value: unknown,
  format: MetricDefinition["format"] = "number",
  fallback = "-"
) {
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

export function formatStatusLabel(status: string | null | undefined) {
  if (status === "completed") return "완료";
  if (status === "failed_nan") return "실패";
  if (status === "failed_quality_gate") return "기준 미달";
  return status ?? "-";
}

export function formatRoleLabel(role: string | null | undefined) {
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

export function formatModelLabel(value: unknown) {
  if (value === "patchtst") return "PatchTST";
  if (value === "cnn_lstm") return "CNN-LSTM";
  if (value === "line_band_composite") return "결합 방식 실험";
  return formatValue(value);
}

export function formatFeatureSet(value: unknown) {
  if (value === "full_features") return "전체 피처";
  if (value === "price_volatility_volume") return "가격·변동성·거래량";
  return formatValue(value);
}

export function formatConfigLabel(key: string) {
  return CONFIG_LABELS[key] ?? key;
}

export function extractErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 설정과 백엔드 상태를 확인해주세요.";
    }
    return error.message;
  }
  return fallback;
}

export function formatSignedNumber(value: number | null, digits = 4) {
  if (value == null) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatValue(value, digits)}`;
}

export function formatSignedPctPoint(value: number | null) {
  if (value == null) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatValue(value, 1)}%p`;
}

export function formatComparisonDiff(value: number, format: MetricDefinition["format"]) {
  if (format === "rate" || format === "pct_point") {
    return formatSignedPctPoint(value * 100);
  }
  return formatSignedNumber(value);
}
