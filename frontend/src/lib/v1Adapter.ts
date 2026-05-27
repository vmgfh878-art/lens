// v1 serving parquet (line_score / band_lower,band_upper) → legacy PredictionResult 형태로 변환.
// 화면 / Chart 는 legacy 스키마를 그대로 사용하므로 변환 layer 가 필요하다.
// 변환 핵심: safe_line_score (수익률 단위) → asof 종가 × (1 + score) 로 가격 환산.

import type {
  PredictionResult,
  PriceBar,
  ProductBandHistoryPoint,
  ProductLineHistoryPoint,
  V1BandPredictionResult,
  V1LinePredictionResult,
} from "@/api/client";

import { addBusinessDays, addDays, isValidDate, sortPriceRows, sortUniqueByAsofDate } from "./dateUtils";
import { finiteOrNull } from "./formatters";
import { PRODUCT_SLOT_BY_ID } from "./productSlots";

export function normalizeBandValueToPrice(value: number, baseClose: number | null) {
  if (Math.abs(value) < 1 && baseClose != null) {
    return baseClose * (1 + value);
  }
  if (Math.abs(value) < 1) {
    return null;
  }
  return value;
}

export function findCloseOnOrBefore(rows: PriceBar[], date: string) {
  let selected: number | null = null;
  for (const row of rows) {
    if (!isValidDate(row.date) || row.date > date) {
      continue;
    }
    if (Number.isFinite(row.close)) {
      selected = row.close;
    }
  }
  return selected;
}

export function findTradingHorizonDate(rows: PriceBar[], asofDate: string, horizon: number) {
  const dates = sortPriceRows(rows).map((row) => row.date);
  let baseIndex = -1;
  for (let index = 0; index < dates.length; index += 1) {
    if (dates[index] <= asofDate) {
      baseIndex = index;
    } else {
      break;
    }
  }

  if (baseIndex >= 0) {
    const targetIndex = baseIndex + horizon;
    if (targetIndex < dates.length) {
      return dates[targetIndex];
    }
  }

  return addBusinessDays(asofDate, horizon);
}

function countTradingSessionsToLatest(rows: PriceBar[], asofDate: string, latestPriceDate: string) {
  const dates = sortPriceRows(rows).map((row) => row.date);
  return dates.filter((date) => date > asofDate && date <= latestPriceDate).length;
}

function resolveFutureTradingDate(
  rows: PriceBar[],
  asofDate: string,
  horizon: number,
  latestPriceDate: string | null,
  futureTradingDates: string[]
) {
  if (latestPriceDate && futureTradingDates.length > 0) {
    const sessionsToLatest = countTradingSessionsToLatest(rows, asofDate, latestPriceDate);
    const targetIndex = futureTradingDates.length - 1 - sessionsToLatest;
    if (targetIndex >= 0 && targetIndex < futureTradingDates.length) {
      return futureTradingDates[targetIndex];
    }
  }
  return findTradingHorizonDate(rows, asofDate, horizon);
}

export function buildLineHistoryFromV1(
  response: V1LinePredictionResult,
  priceRows: PriceBar[]
): ProductLineHistoryPoint[] {
  const slot = PRODUCT_SLOT_BY_ID["line-1d"];
  const rows: ProductLineHistoryPoint[] = [];
  response.data.forEach((row) => {
    const returnValue = finiteOrNull(row.safe_line_score) ?? finiteOrNull(row.line_score);
    const baseClose = findCloseOnOrBefore(priceRows, row.asof_date);
    if (!isValidDate(row.asof_date) || returnValue == null || baseClose == null) {
      return;
    }
    rows.push({
      asof_date: row.asof_date,
      forecast_date: findTradingHorizonDate(priceRows, row.asof_date, slot.horizon ?? 5),
      display_horizon: slot.horizon ?? 5,
      value: baseClose * (1 + returnValue),
      run_id: row.model_id ?? slot.runId ?? slot.sourceCp ?? "line-serving",
    });
  });
  return sortUniqueByAsofDate(rows);
}

export function buildLinePredictionFromV1(
  response: V1LinePredictionResult,
  priceRows: PriceBar[],
  futureTradingDates: string[] = []
): PredictionResult | null {
  const slot = PRODUCT_SLOT_BY_ID["line-1d"];
  const sortedRows = [...response.data]
    .filter((row) => isValidDate(row.asof_date))
    .sort((left, right) => left.asof_date.localeCompare(right.asof_date));
  const latest = sortedRows.at(-1);
  if (!latest) {
    return null;
  }

  const horizon = slot.horizon ?? 5;
  const latestPriceDate = priceRows.length > 0 ? priceRows[priceRows.length - 1]?.date ?? null : null;
  const futureRows = sortedRows
    .map((row) => {
      const returnValue = finiteOrNull(row.safe_line_score) ?? finiteOrNull(row.line_score);
      const baseClose = findCloseOnOrBefore(priceRows, row.asof_date);
      if (returnValue == null || baseClose == null) {
        return null;
      }
      return {
        asof_date: row.asof_date,
        forecast_date: resolveFutureTradingDate(
          priceRows,
          row.asof_date,
          horizon,
          latestPriceDate,
          futureTradingDates
        ),
        line_value: baseClose * (1 + returnValue),
        model_id: row.model_id ?? null,
        source_cp: row.source_cp ?? null,
      };
    })
    .filter(
      (
        row
      ): row is {
        asof_date: string;
        forecast_date: string;
        line_value: number;
        model_id: string | null;
        source_cp: string | null;
      } => row !== null && isValidDate(row.forecast_date)
    )
    .filter((row) => {
      return latestPriceDate ? row.forecast_date > latestPriceDate : true;
    })
    .sort((left, right) => left.forecast_date.localeCompare(right.forecast_date));

  const dedupedFutureRows = Array.from(
    futureRows.reduce((map, row) => {
      map.set(row.forecast_date, row);
      return map;
    }, new Map<string, (typeof futureRows)[number]>()).values()
  ).sort((left, right) => left.forecast_date.localeCompare(right.forecast_date));

  const selectedFutureRows = dedupedFutureRows.slice(-horizon);
  if (selectedFutureRows.length === 0) {
    return null;
  }

  return {
    ticker: response.ticker,
    model_name: slot.modelName,
    timeframe: "1D",
    horizon,
    asof_date: latest.asof_date,
    decision_time: latest.asof_date,
    run_id: latest.model_id ?? response.model_id ?? slot.runId ?? slot.sourceCp ?? "line-serving",
    model_ver: slot.version ?? "v1",
    signal: "HOLD",
    forecast_dates: selectedFutureRows.map((row) => row.forecast_date),
    upper_band_series: [],
    lower_band_series: [],
    conservative_series: selectedFutureRows.map((row) => row.line_value),
    line_series: selectedFutureRows.map((row) => row.line_value),
    band_quantile_low: null,
    band_quantile_high: null,
    meta: {
      role: "line_model",
      layer: "line",
      source_cp: latest.source_cp ?? response.source_cp ?? slot.sourceCp,
      model_id: latest.model_id ?? response.model_id ?? slot.modelId,
      value_unit: "price",
      raw_value_unit: "return",
      transform: "asof_close_times_one_plus_h5_safe_line_score",
      display_horizon: "h5",
      refresh_policy: "active_serving_parquet",
    },
  };
}

export function selectBandDisplayRows(response: V1BandPredictionResult, preferredHorizon: number) {
  const byAsof = new Map<string, V1BandPredictionResult["data"]>();
  response.data.forEach((row) => {
    if (!isValidDate(row.asof_date)) {
      return;
    }
    const rows = byAsof.get(row.asof_date) ?? [];
    rows.push(row);
    byAsof.set(row.asof_date, rows);
  });

  return Array.from(byAsof.entries())
    .map(([, rows]) => {
      const preferred = rows.find((row) => row.horizon_step === preferredHorizon);
      const fallback = rows.sort((left, right) => right.horizon_step - left.horizon_step)[0];
      return preferred ?? fallback ?? null;
    })
    .filter((row): row is V1BandPredictionResult["data"][number] => row !== null);
}

export function buildBandHistoryFromV1(
  response: V1BandPredictionResult,
  preferredHorizon: number,
  priceRows: PriceBar[]
): ProductBandHistoryPoint[] {
  const mappedRows: Array<ProductBandHistoryPoint | null> = selectBandDisplayRows(response, preferredHorizon).map((row) => {
      const rawLower = finiteOrNull(row.band_lower);
      const rawUpper = finiteOrNull(row.band_upper);
      const baseClose = findCloseOnOrBefore(priceRows, row.asof_date);
      const lower = rawLower == null ? null : normalizeBandValueToPrice(rawLower, baseClose);
      const upper = rawUpper == null ? null : normalizeBandValueToPrice(rawUpper, baseClose);
      if (!isValidDate(row.asof_date) || lower == null || upper == null) {
        return null;
      }
      return {
        asof_date: row.asof_date,
        forecast_date:
          row.forecast_date ?? findTradingHorizonDate(priceRows, row.asof_date, row.horizon_step ?? preferredHorizon),
        display_horizon: row.horizon_step,
        lower,
        upper,
        run_id: row.model_id ?? response.model_id ?? response.source_cp ?? response.slot,
      };
    });

  const rows: ProductBandHistoryPoint[] = mappedRows.filter(
    (row): row is ProductBandHistoryPoint => row !== null
  );

  return sortUniqueByAsofDate(rows);
}

export function buildBandPredictionFromV1(
  response: V1BandPredictionResult,
  timeframe: "1D" | "1W",
  preferredHorizon: number,
  priceRows: PriceBar[]
): PredictionResult | null {
  const sortedRows = [...response.data]
    .filter((row) => isValidDate(row.asof_date))
    .sort((left, right) => {
      const byDate = left.asof_date.localeCompare(right.asof_date);
      return byDate !== 0 ? byDate : left.horizon_step - right.horizon_step;
    });
  const latestAsof = sortedRows.at(-1)?.asof_date;
  if (!latestAsof) {
    return null;
  }
  const latestRows = sortedRows
    .filter((row) => row.asof_date === latestAsof)
    .sort((left, right) => left.horizon_step - right.horizon_step);
  const rows = latestRows.filter((row) => row.horizon_step <= preferredHorizon);
  const forecastDates = rows.map(
    (row) =>
      row.forecast_date ?? addDays(row.asof_date, timeframe === "1W" ? row.horizon_step * 7 : row.horizon_step)
  );
  const baseClose = findCloseOnOrBefore(priceRows, latestAsof);
  const upper = rows.map((row) => {
    const value = finiteOrNull(row.band_upper);
    return value == null ? null : normalizeBandValueToPrice(value, baseClose);
  });
  const lower = rows.map((row) => {
    const value = finiteOrNull(row.band_lower);
    return value == null ? null : normalizeBandValueToPrice(value, baseClose);
  });
  if (forecastDates.length === 0 || upper.some((value) => value == null) || lower.some((value) => value == null)) {
    return null;
  }
  const slot = timeframe === "1W" ? PRODUCT_SLOT_BY_ID["band-1w"] : PRODUCT_SLOT_BY_ID["band-1d"];
  return {
    ticker: response.ticker,
    model_name: slot.modelName,
    timeframe,
    horizon: preferredHorizon,
    asof_date: latestAsof,
    decision_time: latestAsof,
    run_id: response.model_id ?? slot.runId ?? slot.sourceCp ?? response.slot,
    model_ver: slot.version ?? "v1",
    signal: "HOLD",
    forecast_dates: forecastDates,
    upper_band_series: upper as number[],
    lower_band_series: lower as number[],
    conservative_series: [],
    line_series: [],
    band_quantile_low: null,
    band_quantile_high: null,
    meta: {
      role: "band_model",
      layer: "band",
      source_cp: response.source_cp ?? slot.sourceCp,
      model_id: response.model_id ?? slot.modelId,
      value_unit: "price",
      raw_value_unit: baseClose == null ? "price" : "auto_normalized_to_price",
      refresh_policy: "auto_parquet",
    },
  };
}
