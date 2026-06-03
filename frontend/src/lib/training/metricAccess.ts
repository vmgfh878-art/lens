// AiRunDetail.test_metrics / val_metrics에서 키 fallback으로 값을 꺼내는 헬퍼.
// experimentBuilder + TrainingView 본문(ExperimentDetail / loadRuns) 양쪽이 사용.

import type { AiRunDetail } from "@/api/client";
import {
  BAND_METRICS,
  LINE_METRICS,
  MetricDefinition,
  ProductSlot,
} from "@/lib/training/constants";
import { formatMetric } from "@/lib/training/formatters";

export function getMetricByKeys(metrics: Record<string, unknown> | null | undefined, keys: string[]) {
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

export function getMetricText(
  detail: AiRunDetail | null,
  definition: MetricDefinition,
  fallback: string,
  source: "test" | "val" = "test"
) {
  const metrics = source === "test" ? detail?.test_metrics : detail?.val_metrics;
  return formatMetric(getMetricByKeys(metrics, definition.keys), definition.format, fallback);
}

export function getMetricNumber(detail: AiRunDetail | null, definition: MetricDefinition, source: "test" | "val" = "test") {
  const metrics = source === "test" ? detail?.test_metrics : detail?.val_metrics;
  const value = getMetricByKeys(metrics, definition.keys);
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function getMetricNumberFromStoredEvaluation(detail: AiRunDetail | null, definition: MetricDefinition) {
  return getMetricNumber(detail, definition, "test") ?? getMetricNumber(detail, definition, "val");
}

export function hasStoredEvaluationMetrics(detail: AiRunDetail | null, definitions: MetricDefinition[]) {
  return definitions.some((definition) => getMetricNumberFromStoredEvaluation(detail, definition) != null);
}

export function getProductMetricDefinitions(slot: ProductSlot) {
  if (slot.kind === "band") {
    return BAND_METRICS;
  }
  if (slot.kind === "line") {
    return LINE_METRICS;
  }
  return [];
}
