// Backtest 시뮬레이션에서 prediction 으로부터 핵심 신호값 추출.

import type { PredictionResult } from "@/api/client";

function getLastFinite(values: number[] | null | undefined) {
  if (!values) {
    return null;
  }
  for (let index = values.length - 1; index >= 0; index -= 1) {
    if (Number.isFinite(values[index])) {
      return values[index];
    }
  }
  return null;
}

export function getConservativeValue(prediction: PredictionResult) {
  return getLastFinite(
    prediction.conservative_series?.length ? prediction.conservative_series : prediction.line_series
  );
}

export function getWorstLowerBandValue(prediction: PredictionResult) {
  const values = prediction.lower_band_series.filter(Number.isFinite);
  return values.length > 0 ? Math.min(...values) : null;
}

export function getHighestUpperBandValue(prediction: PredictionResult) {
  const values = prediction.upper_band_series.filter(Number.isFinite);
  return values.length > 0 ? Math.max(...values) : null;
}

export function getBandWidthValue(prediction: PredictionResult) {
  const lower = getWorstLowerBandValue(prediction);
  const upper = getHighestUpperBandValue(prediction);
  return lower != null && upper != null ? upper - lower : null;
}

export function median(values: number[]) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

export function percentileRank(value: number, values: number[]) {
  const usable = values.filter(Number.isFinite);
  if (usable.length === 0) {
    return null;
  }
  const lowerOrEqual = usable.filter((candidate) => candidate <= value).length;
  return lowerOrEqual / usable.length;
}
