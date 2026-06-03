// CP233 — StockView에서 분리한 순수 헬퍼 박제.

import { afterEach, describe, expect, it, vi } from "vitest";

import type { PriceBar } from "@/api/client";
import { PRICE_LOOKBACK_LIMIT_1D, PRICE_LOOKBACK_LIMIT_1W } from "@/lib/constants";

import {
  buildAiState,
  fetchPriceHistory,
  getChangePercent,
  getLastFinite,
  getLastPrice,
  getPriceLookbackDays,
} from "./helpers";

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return {
    ...actual,
    fetchPrices: vi.fn(async (_ticker: string, _params: unknown) => ({
      data: { data: [] as PriceBar[] },
    })),
  };
});

afterEach(() => {
  vi.clearAllMocks();
});

function makeBar(date: string, close: number): PriceBar {
  return { date, open: close, high: close, low: close, close, volume: null };
}

describe("getLastPrice", () => {
  it("빈 배열 → null", () => {
    expect(getLastPrice([])).toBeNull();
  });
  it("마지막 bar 반환", () => {
    const bars = [makeBar("2024-01-01", 100), makeBar("2024-01-02", 101)];
    expect(getLastPrice(bars)).toBe(bars[1]);
  });
});

describe("getChangePercent", () => {
  it("길이 < 2 → null", () => {
    expect(getChangePercent([])).toBeNull();
    expect(getChangePercent([makeBar("2024-01-01", 100)])).toBeNull();
  });
  it("previous = 0 → null", () => {
    expect(getChangePercent([makeBar("2024-01-01", 0), makeBar("2024-01-02", 100)])).toBeNull();
  });
  it("정상 % 계산", () => {
    expect(getChangePercent([makeBar("2024-01-01", 100), makeBar("2024-01-02", 105)])).toBeCloseTo(5, 6);
    expect(getChangePercent([makeBar("2024-01-01", 100), makeBar("2024-01-02", 95)])).toBeCloseTo(-5, 6);
  });
});

describe("getLastFinite", () => {
  it("null/undefined → null", () => {
    expect(getLastFinite(null)).toBeNull();
    expect(getLastFinite(undefined)).toBeNull();
  });
  it("뒤에서 첫 finite 반환", () => {
    expect(getLastFinite([1, 2, 3])).toBe(3);
    expect(getLastFinite([1, 2, Number.NaN])).toBe(2);
    expect(getLastFinite([1, Number.NaN, Number.NaN])).toBe(1);
  });
  it("전부 NaN → null", () => {
    expect(getLastFinite([Number.NaN, Number.NaN])).toBeNull();
    expect(getLastFinite([])).toBeNull();
  });
});

describe("getPriceLookbackDays", () => {
  it("1W → PRICE_LOOKBACK_LIMIT_1W", () => {
    expect(getPriceLookbackDays("1W")).toBe(PRICE_LOOKBACK_LIMIT_1W);
  });
  it("1D / 1M → PRICE_LOOKBACK_LIMIT_1D", () => {
    expect(getPriceLookbackDays("1D")).toBe(PRICE_LOOKBACK_LIMIT_1D);
    expect(getPriceLookbackDays("1M")).toBe(PRICE_LOOKBACK_LIMIT_1D);
  });
});

describe("buildAiState", () => {
  it("1M → disabled + 메시지 박제", () => {
    expect(buildAiState("1M")).toEqual({
      kind: "disabled",
      message: "월간 화면은 현재 가격 전용입니다.",
    });
  });
  it("1D / 1W → empty + 메시지 박제", () => {
    expect(buildAiState("1D")).toEqual({
      kind: "empty",
      message: "저장된 최신 예측 결과가 아직 없습니다.",
    });
    expect(buildAiState("1W")).toEqual({
      kind: "empty",
      message: "저장된 최신 예측 결과가 아직 없습니다.",
    });
  });
});

describe("fetchPriceHistory", () => {
  it("fullHistory=false → 단일 window fetchPrices 호출 + sort", async () => {
    const { fetchPrices } = await import("@/api/client");
    const mocked = vi.mocked(fetchPrices);
    mocked.mockResolvedValueOnce({
      data: { data: [makeBar("2024-01-02", 101), makeBar("2024-01-01", 100)] },
    } as Awaited<ReturnType<typeof fetchPrices>>);

    const result = await fetchPriceHistory("AAPL", "1D");
    expect(mocked).toHaveBeenCalledTimes(1);
    const call = mocked.mock.calls[0];
    expect(call[0]).toBe("AAPL");
    expect(call[1]).toMatchObject({ timeframe: "1D" });
    expect(result.map((row) => row.date)).toEqual(["2024-01-01", "2024-01-02"]);
  });
  it("fullHistory=true → 여러 window 병렬 호출", async () => {
    const { fetchPrices } = await import("@/api/client");
    const mocked = vi.mocked(fetchPrices);
    mocked.mockResolvedValue({
      data: { data: [makeBar("2024-01-01", 100)] },
    } as Awaited<ReturnType<typeof fetchPrices>>);

    await fetchPriceHistory("AAPL", "1D", true);
    expect(mocked.mock.calls.length).toBeGreaterThan(1);
  });
});
