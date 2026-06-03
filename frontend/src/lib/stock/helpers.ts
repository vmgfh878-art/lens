// StockView 순수 헬퍼 + 얇은 fetch wrapper. 부작용 없음 (fetchPriceHistory는 fetchPrices 위임).

import { DisplayTimeframe, fetchPrices, PriceBar } from "@/api/client";
import { PRICE_LOOKBACK_LIMIT_1D, PRICE_LOOKBACK_LIMIT_1W } from "@/lib/constants";
import { buildDefaultPriceWindow, buildFullPriceWindows, sortPriceRows } from "@/lib/dateUtils";

export interface AiState {
  kind: "loading" | "ready" | "disabled" | "empty";
  message: string;
}

export const FULL_PRICE_HISTORY_START_YEAR = 2015;

// CP215 — 1D/1W timeframe 별 가격 lookback (calendar days).
export function getPriceLookbackDays(timeframe: DisplayTimeframe) {
  if (timeframe === "1W") {
    return PRICE_LOOKBACK_LIMIT_1W;
  }
  // 1D + 1M 모두 일봉 단위 fetch. 1M 은 별도 표시 모드이지만 lookback 은 1D 와 동일.
  return PRICE_LOOKBACK_LIMIT_1D;
}

export function getLastPrice(rows: PriceBar[]) {
  return rows.length > 0 ? rows[rows.length - 1] : null;
}

export function getChangePercent(rows: PriceBar[]) {
  if (rows.length < 2) {
    return null;
  }
  const latest = rows[rows.length - 1].close;
  const previous = rows[rows.length - 2].close;
  if (!previous) {
    return null;
  }
  return ((latest - previous) / previous) * 100;
}

export function getLastFinite(values: number[] | null | undefined) {
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

export function buildAiState(timeframe: DisplayTimeframe): AiState {
  if (timeframe === "1M") {
    return {
      kind: "disabled",
      message: "월간 화면은 현재 가격 전용입니다.",
    };
  }
  return {
    kind: "empty",
    message: "저장된 최신 예측 결과가 아직 없습니다.",
  };
}

export async function fetchPriceHistory(ticker: string, timeframe: DisplayTimeframe, fullHistory = false) {
  if (!fullHistory) {
    const window = buildDefaultPriceWindow(getPriceLookbackDays(timeframe));
    const response = await fetchPrices(ticker, {
      timeframe,
      start: window.start,
      end: window.end,
    });
    return sortPriceRows(response.data.data);
  }

  const responses = await Promise.all(
    buildFullPriceWindows(FULL_PRICE_HISTORY_START_YEAR).map((window) =>
      fetchPrices(ticker, {
        timeframe,
        start: window.start,
        end: window.end,
      })
    )
  );
  return sortPriceRows(responses.flatMap((response) => response.data.data));
}
