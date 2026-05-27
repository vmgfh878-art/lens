import type { PredictionResult, PriceBar } from "@/api/client";
import type { ProductSlot } from "@/lib/productSlots";

import {
  countPriceRowsAfter,
  diffCalendarDaysBetween,
  isValidDate,
  uniqueDates,
} from "./dateUtils";

export type SlotFreshness = "fresh" | "delayed" | "stale" | "static" | "deferred" | "empty";

// 슬롯의 갱신 정책 (auto/static/deferred) 과 timeframe 별 기준일 거리 기준으로
// fresh/delayed/stale 을 판정한다.
// - 1D: 가격 거래일 카운트 기준
// - 1W: 달력 일 수 기준
export function getSlotFreshness(
  slot: ProductSlot | null | undefined,
  asofDate: string | null | undefined,
  latestPriceDate: string | null | undefined,
  priceRows: PriceBar[]
): SlotFreshness {
  if (!slot) {
    return "empty";
  }
  if (slot.refreshPolicy === "deferred") {
    return "deferred";
  }
  if (slot.refreshPolicy === "static") {
    return "static";
  }
  if (!isValidDate(asofDate) || !isValidDate(latestPriceDate)) {
    return "empty";
  }

  if (slot.timeframe === "1D") {
    const tradingRowsAfter = countPriceRowsAfter(priceRows, asofDate, latestPriceDate);
    if (tradingRowsAfter == null) {
      return "empty";
    }
    if (slot.staleAfterDays != null && tradingRowsAfter >= slot.staleAfterDays) {
      return "stale";
    }
    if (slot.freshAfterDays != null && tradingRowsAfter <= slot.freshAfterDays) {
      return "fresh";
    }
    return "delayed";
  }

  const ageDays = diffCalendarDaysBetween(asofDate, latestPriceDate);
  if (ageDays == null) {
    return "empty";
  }
  if (slot.staleAfterDays != null && ageDays >= slot.staleAfterDays) {
    return "stale";
  }
  if (slot.freshAfterDays != null && ageDays <= slot.freshAfterDays) {
    return "fresh";
  }
  return "delayed";
}

export function getFutureForecastDates(
  prediction: PredictionResult | null | undefined,
  latestPriceDate: string | null | undefined
) {
  if (!prediction || !isValidDate(latestPriceDate)) {
    return [];
  }
  return uniqueDates(
    (prediction.forecast_dates ?? []).filter((date) => isValidDate(date) && date > latestPriceDate)
  );
}

export function formatFreshnessStatus(status: SlotFreshness) {
  const labels: Record<SlotFreshness, string> = {
    fresh: "fresh",
    delayed: "delayed",
    stale: "stale",
    static: "static",
    deferred: "deferred",
    empty: "데이터 없음",
  };
  return labels[status];
}

export function getFreshnessClass(status: SlotFreshness) {
  if (status === "fresh") return "fresh";
  if (status === "stale") return "stale";
  if (status === "static" || status === "deferred") return "static";
  if (status === "delayed") return "delayed";
  return "empty";
}
