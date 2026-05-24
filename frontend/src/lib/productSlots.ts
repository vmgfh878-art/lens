import type { DisplayTimeframe, PredictionTimeframe } from "@/api/client";

export type ProductSlotId = "line-1d" | "band-1d" | "line-1w" | "band-1w";
export type ProductSlotKind = "line" | "band";
export type ProductSlotStatus = "active" | "deferred";

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
  summary: string;
  endpoint: string | null;
}

export const PRODUCT_SLOT_BY_ID: Record<ProductSlotId, ProductSlot> = {
  "line-1d": {
    id: "line-1d",
    kind: "line",
    title: "1D 보수적 예측선",
    displayName: "보수적 예측선 v1",
    modelName: "Line v2",
    version: "v1",
    timeframe: "1D",
    horizon: 5,
    sourceCp: "CP175",
    modelId: "line_b_cp175_beta5_frozen_eval",
    runId: "line_b_cp175_beta5_frozen_eval",
    status: "active",
    summary: "1D 방향과 하방 위험을 보수적으로 보기 위한 예측선입니다.",
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
    summary: "1D 예상 변동 범위를 보여주는 리스크 참고 밴드입니다.",
    endpoint: "/api/v1/predictions/band/1d/{ticker}",
  },
  "line-1w": {
    id: "line-1w",
    kind: "line",
    title: "1W 보수적 예측선",
    displayName: "1W 보수적 예측선",
    modelName: "준비 중",
    version: null,
    timeframe: "1W",
    horizon: null,
    sourceCp: null,
    modelId: null,
    runId: null,
    status: "deferred",
    summary: "1W 보수적 예측선은 아직 제품 슬롯에 연결하지 않았습니다.",
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
    summary: "1W 예상 변동 범위를 보여주는 주간 리스크 참고 밴드입니다.",
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
