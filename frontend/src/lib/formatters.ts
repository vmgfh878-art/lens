// 가격/지표/예측 값을 사람이 읽기 좋은 형태로 변환하는 순수 함수 모음.
// 모든 함수는 null / undefined / NaN 입력을 "-" 또는 null 로 안전하게 처리한다.

export function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatVolume(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatPercent(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, digits)}%`;
}

export function formatRatio(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  if (Math.abs(value) <= 1) {
    return `${formatNumber(value * 100, 1)}%`;
  }
  return formatNumber(value, 2);
}

export function formatSignedPercentPoint(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}%`;
}

export function getMetaString(meta: Record<string, unknown> | undefined, key: string) {
  const value = meta?.[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

// 0~1 범위와 0~100 범위 둘 다 들어올 수 있는 RSI 정규화.
export function normalizeRsi(value: number) {
  return value >= 0 && value <= 1 ? value * 100 : value;
}

export function ratioToPercent(value: number) {
  return value * 100;
}

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function finiteOrNull(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// 비율 (0~1) 를 퍼센트 (부호 포함) 로 표시. e.g., 0.05 → "+5.00%"
export function formatRatioAsPercent(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return formatPercent(value * 100, digits);
}

// 비율 (0~1) 를 퍼센트 (부호 없이) 로 표시. e.g., 0.05 → "5.00%"
export function formatUnsignedRatioAsPercent(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${formatNumber(value * 100, digits)}%`;
}

// compact 표기. e.g., 12345 → "1.2만"
export function formatCompact(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}
