import type { DisplayTimeframe, PredictionTimeframe } from "@/api/client";

export type ProductSlotId = "line-1d" | "band-1d" | "line-1w" | "band-1w";
export type ProductSlotKind = "line" | "band";
export type ProductSlotStatus = "active" | "deferred";
export type ProductSlotRefreshPolicy = "static" | "auto" | "deferred";

export interface ProductSlot {
  id: ProductSlotId;
  kind: ProductSlotKind;
  title: string;
  displayName: string;
  modelName: string;
  version: string | null;
  timeframe: PredictionTimeframe;
  horizon: number | null;
  sourceCp: string | null;
  modelId: string | null;
  runId: string | null;
  status: ProductSlotStatus;
  refreshPolicy: ProductSlotRefreshPolicy;
  freshAfterDays: number | null;
  staleAfterDays: number | null;
  summary: string;
  endpoint: string | null;
}

export const PRODUCT_SLOT_BY_ID: Record<ProductSlotId, ProductSlot> = {
  "line-1d": {
    id: "line-1d",
    kind: "line",
    title: "1D 보수적 기준선",
    displayName: "보수적 기준선 v1",
    modelName: "F4 β=4 ensemble",
    version: "v1",
    timeframe: "1D",
    horizon: 5,
    sourceCp: "CP208Z/CP209 F4 β=4",
    modelId: "cp210_F4_b4_ensemble_mean",
    runId: "cp210_F4_b4_ensemble_mean",
    status: "active",
    refreshPolicy: "auto",
    freshAfterDays: 1,
    staleAfterDays: 5,
    summary: "활성 1D 보수적 기준선입니다.\n앞으로 5거래일의 보수적 도착가를 추정해 차트에 표시합니다.",
    endpoint: "/api/v1/predictions/line/{ticker}",
  },
  "band-1d": {
    id: "band-1d",
    kind: "band",
    title: "1D AI 밴드",
    displayName: "AI 밴드 v1",
    modelName: "TiDE",
    version: "v1",
    timeframe: "1D",
    horizon: 5,
    sourceCp: "CP153",
    modelId: "tide-1D-ea54dcae654d",
    runId: "tide-1D-ea54dcae654d",
    status: "active",
    refreshPolicy: "auto",
    freshAfterDays: 1,
    staleAfterDays: 5,
    summary: "자동 갱신되는 1D AI 밴드입니다.\n앞으로 5거래일의 예상 변동 범위를 차트에 표시합니다.",
    endpoint: "/api/v1/predictions/band/1d/{ticker}",
  },
  "line-1w": {
    id: "line-1w",
    kind: "line",
    title: "1W 보수적 기준선",
    displayName: "1W 보수적 기준선",
    modelName: "준비 중",
    version: null,
    timeframe: "1W",
    horizon: null,
    sourceCp: null,
    modelId: null,
    runId: null,
    status: "deferred",
    refreshPolicy: "deferred",
    freshAfterDays: null,
    staleAfterDays: null,
    summary: "1W 보수적 기준선은 v1에서 제공하지 않습니다.\n주간 단위에서 안정 성능이 안 나와 제외했습니다.",
    endpoint: null,
  },
  "band-1w": {
    id: "band-1w",
    kind: "band",
    title: "1W AI 밴드",
    displayName: "1W AI 밴드 v1",
    modelName: "TiDE",
    version: "v1",
    timeframe: "1W",
    horizon: 4,
    sourceCp: "CP178",
    modelId: "tide_s60_q10_q90_param",
    runId: "tide_s60_q10_q90_param",
    status: "active",
    refreshPolicy: "auto",
    freshAfterDays: 7,
    staleAfterDays: 14,
    summary: "자동 갱신되는 1W AI 밴드입니다.\n앞으로 4주의 예상 변동 범위를 차트에 표시합니다.",
    endpoint: "/api/v1/predictions/band/1w/{ticker}",
  },
};

export const PRODUCT_SLOTS = [
  PRODUCT_SLOT_BY_ID["line-1d"],
  PRODUCT_SLOT_BY_ID["band-1d"],
  PRODUCT_SLOT_BY_ID["line-1w"],
  PRODUCT_SLOT_BY_ID["band-1w"],
] as const;

export const PRODUCT_RUN_IDS = new Set(
  PRODUCT_SLOTS.map((slot) => slot.runId).filter((runId): runId is string => Boolean(runId))
);

export function getProductLineSlot(timeframe: DisplayTimeframe) {
  if (timeframe === "1D") {
    return PRODUCT_SLOT_BY_ID["line-1d"];
  }
  if (timeframe === "1W") {
    return PRODUCT_SLOT_BY_ID["line-1w"];
  }
  return null;
}

export function getProductBandSlot(timeframe: DisplayTimeframe) {
  if (timeframe === "1D") {
    return PRODUCT_SLOT_BY_ID["band-1d"];
  }
  if (timeframe === "1W") {
    return PRODUCT_SLOT_BY_ID["band-1w"];
  }
  return null;
}
