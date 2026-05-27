// 예측 overlay 검증 / 변환 helper.
// Chart 가 그릴 line / band 데이터의 유효성, legacy 여부, IndicatorPanel 시리즈 빌드 등을 다룬다.

import type {
  IndicatorPoint,
  PredictionResult,
  PriceBar,
  ProductBandHistoryPoint,
} from "@/api/client";
import type { IndicatorChartPoint, IndicatorChartSeries } from "@/components/IndicatorPanel";

import { sortUniqueByDate } from "./dateUtils";
import { getMetaString, isFiniteNumber } from "./formatters";
import { buildFieldPoints, IndicatorDefinition, IndicatorId } from "./indicators";

export interface PredictionOverlayCheck {
  ok: boolean;
  message: string | null;
}

export function isLegacyPrediction(prediction: PredictionResult | null) {
  if (!prediction) {
    return false;
  }
  const role = getMetaString(prediction.meta, "role");
  const deprecated = prediction.meta?.deprecated_for_phase1_product_contract;
  return (
    prediction.model_name.includes("composite") ||
    role === "composite_model" ||
    deprecated === true ||
    deprecated === "true"
  );
}

export function checkLineOverlay(prediction: PredictionResult): PredictionOverlayCheck {
  const forecastDates = prediction.forecast_dates ?? [];
  const lineValues = prediction.conservative_series?.length
    ? prediction.conservative_series
    : prediction.line_series ?? [];

  if (forecastDates.length === 0) {
    return { ok: false, message: "예측 날짜가 없어 보수적 기준선을 숨겼습니다." };
  }
  if (lineValues.length !== forecastDates.length) {
    return { ok: false, message: "예측 날짜와 시리즈 길이가 맞지 않아 보수적 기준선을 숨겼습니다." };
  }

  const allValues = [...lineValues];
  if (forecastDates.some((date) => !date) || allValues.some((value) => !Number.isFinite(value))) {
    return { ok: false, message: "예측 데이터에 표시할 수 없는 값이 있어 보수적 기준선을 숨겼습니다." };
  }

  return { ok: true, message: null };
}

export function checkBandOverlay(prediction: PredictionResult): PredictionOverlayCheck {
  const forecastDates = prediction.forecast_dates ?? [];

  if (forecastDates.length === 0) {
    return { ok: false, message: "예측 날짜가 없어 AI 밴드를 숨겼습니다." };
  }
  if (
    prediction.upper_band_series.length !== forecastDates.length ||
    prediction.lower_band_series.length !== forecastDates.length
  ) {
    return { ok: false, message: "예측 날짜와 밴드 길이가 맞지 않아 AI 밴드를 숨겼습니다." };
  }

  const allValues = [...prediction.upper_band_series, ...prediction.lower_band_series];
  if (forecastDates.some((date) => !date) || allValues.some((value) => !Number.isFinite(value))) {
    return { ok: false, message: "AI 밴드 데이터에 표시할 수 없는 값이 있어 숨겼습니다." };
  }

  return { ok: true, message: null };
}

// PredictionResult 의 meta 가 진짜 band 모델 (price 단위 band) 인지 판정.
// composite 또는 degenerate band 는 화면에서 가리는 데 사용.
export function isActualBandPrediction(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return false;
  }

  const role = getMetaString(prediction.meta, "role");
  const bandFieldsPolicy = getMetaString(prediction.meta, "band_fields_policy");
  const bandSavedInCp140 = prediction.meta?.band_saved_in_cp140;

  if (role !== "band_model") {
    return false;
  }

  if (bandSavedInCp140 === false || bandSavedInCp140 === "false") {
    return false;
  }

  return bandFieldsPolicy !== "schema_required_degenerate_equal_to_line";
}

export function getPredictionLineValues(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return [];
  }
  return prediction.conservative_series?.length
    ? prediction.conservative_series
    : prediction.line_series ?? [];
}

export function isPredictionLineWithinPriceRange(prediction: PredictionResult, rows: PriceBar[]) {
  const values = getPredictionLineValues(prediction).filter(Number.isFinite);
  const priceValues = rows
    .flatMap((row) => [row.low ?? row.close, row.high ?? row.close, row.close])
    .filter(Number.isFinite);
  if (values.length === 0 || priceValues.length < 2) {
    return false;
  }

  const minPrice = Math.min(...priceValues);
  const maxPrice = Math.max(...priceValues);
  const latestPrice = priceValues[priceValues.length - 1] ?? maxPrice;
  const span = Math.max(maxPrice - minPrice, Math.abs(latestPrice) * 0.04, 1);
  const lowerBound = minPrice - span * 0.35;
  const upperBound = maxPrice + span * 0.35;
  return values.every((value) => value >= lowerBound && value <= upperBound);
}

export function buildAiBandWidthPoints(
  history: ProductBandHistoryPoint[],
  latestPrediction: PredictionResult | null
): IndicatorChartPoint[] {
  const historyPoints = history
    .map((row) => {
      if (!row.asof_date || !Number.isFinite(row.upper) || !Number.isFinite(row.lower)) {
        return null;
      }
      return {
        date: row.asof_date,
        value: Math.max(0, row.upper - row.lower),
      };
    })
    .filter((point): point is IndicatorChartPoint => point !== null);

  const latestUpper = latestPrediction?.upper_band_series?.[0];
  const latestLower = latestPrediction?.lower_band_series?.[0];
  if (latestPrediction?.asof_date && isFiniteNumber(latestUpper) && isFiniteNumber(latestLower)) {
    historyPoints.push({
      date: latestPrediction.asof_date,
      value: Math.max(0, latestUpper - latestLower),
    });
  }

  return sortUniqueByDate(historyPoints);
}

function getLastPoint(points: IndicatorChartPoint[]) {
  return points.length > 0 ? points[points.length - 1] : null;
}

export function buildVisibleIndicatorSeries(params: {
  selectedIndicators: IndicatorId[];
  indicatorData: IndicatorPoint[];
  indicatorErrorMessage: string | null;
  availableDefinitions: IndicatorDefinition[];
  allowedDates?: Set<string>;
  aiBandWidthPoints: IndicatorChartPoint[];
}): IndicatorChartSeries[] {
  const {
    selectedIndicators,
    indicatorData,
    indicatorErrorMessage,
    availableDefinitions,
    allowedDates,
    aiBandWidthPoints,
  } = params;
  const indicatorEmptyMessage = indicatorErrorMessage ?? "저장된 보조지표 데이터가 없습니다.";

  return availableDefinitions
    .filter((definition) => selectedIndicators.includes(definition.id))
    .map((definition) => {
      if (definition.id === "ai_band_width") {
        const latest = getLastPoint(aiBandWidthPoints)?.value;
        return {
          id: definition.id,
          label: definition.label,
          groupLabel: definition.category,
          points: aiBandWidthPoints,
          color: definition.color,
          baseline: definition.baseline,
          fixedRange: definition.fixedRange,
          latestLabel: definition.formatLatest(latest),
          emptyMessage: "저장된 AI 밴드 폭 이력이 없습니다.",
        };
      }

      const points = buildFieldPoints(indicatorData, definition, allowedDates);
      const latest = getLastPoint(points)?.value;

      return {
        id: definition.id,
        label: definition.label,
        groupLabel: definition.category,
        points,
        color: definition.color,
        baseline: definition.baseline,
        fixedRange: definition.fixedRange,
        latestLabel: definition.formatLatest(latest),
        emptyMessage: indicatorEmptyMessage,
      };
    });
}
