// 이전 실험 detail → "목표 대비 평가" GoalCardData 빌더 + 실패 사유 산출.
// 부작용 없음. TrainingView가 import해 LineModelDetail/BandModelDetail JSX에 사용.

import type { AiRunDetail } from "@/api/client";
import { BAND_METRICS, LINE_METRICS } from "@/lib/training/constants";
import { formatMetric, formatSignedNumber, formatSignedPctPoint } from "@/lib/training/formatters";
import { getExperimentKind } from "@/lib/training/runUtils";
import type { GoalCardData } from "@/lib/training/cardTypes";
import { getMetricNumber, getMetricNumberFromStoredEvaluation } from "@/lib/training/metricAccess";

export function buildLineExperimentCards(detail: AiRunDetail): GoalCardData[] {
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

export function buildBandExperimentCards(detail: AiRunDetail): GoalCardData[] {
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

export function buildExperimentCards(detail: AiRunDetail) {
  return getExperimentKind(detail) === "band" ? buildBandExperimentCards(detail) : buildLineExperimentCards(detail);
}

export function hasDisplayableExperimentMetrics(detail: AiRunDetail) {
  return buildExperimentCards(detail).length >= 2;
}

export function getExperimentFailureReason(detail: AiRunDetail) {
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
