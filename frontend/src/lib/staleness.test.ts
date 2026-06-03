// CP230 characterization — lib/staleness.ts 박제.
// 내부 new Date() 미사용 → 모든 입력 명시 = 결정론.

import { describe, expect, it } from "vitest";

import type { PriceBar } from "@/api/client";
import type { ProductSlot } from "@/lib/productSlots";

import { evaluateBandStaleness, getSlotFreshness } from "./staleness";

function makeSlot(overrides: Partial<ProductSlot> = {}): ProductSlot {
  return {
    timeframe: "1D",
    refreshPolicy: "auto",
    freshAfterDays: 1,
    staleAfterDays: 5,
    ...overrides,
  } as ProductSlot;
}

function makePrice(date: string): PriceBar {
  return {
    date,
    open: 100,
    high: 100,
    low: 100,
    close: 100,
    volume: 0,
  } as PriceBar;
}

describe("getSlotFreshness", () => {
  it("slot null → 'empty'", () => {
    expect(getSlotFreshness(null, "2026-06-01", "2026-06-02", [])).toBe(
      "empty",
    );
  });
  it("refreshPolicy 'deferred' → 'deferred'", () => {
    expect(
      getSlotFreshness(
        makeSlot({ refreshPolicy: "deferred" }),
        "2026-06-01",
        "2026-06-02",
        [],
      ),
    ).toBe("deferred");
  });
  it("refreshPolicy 'static' → 'static'", () => {
    expect(
      getSlotFreshness(
        makeSlot({ refreshPolicy: "static" }),
        "2026-06-01",
        "2026-06-02",
        [],
      ),
    ).toBe("static");
  });
  it("asof invalid → 'empty'", () => {
    expect(
      getSlotFreshness(makeSlot(), "invalid-date", "2026-06-02", []),
    ).toBe("empty");
  });
  it("latestPrice invalid → 'empty'", () => {
    expect(getSlotFreshness(makeSlot(), "2026-06-01", null, [])).toBe("empty");
  });

  it("1D + asof=price 같은 날 + priceRow 있음 → fresh (tradingRows 0)", () => {
    const slot = makeSlot({
      timeframe: "1D",
      freshAfterDays: 1,
      staleAfterDays: 5,
    });
    const prices = [makePrice("2026-06-02")];
    expect(getSlotFreshness(slot, "2026-06-02", "2026-06-02", prices)).toBe(
      "fresh",
    );
  });

  it("1W 달력일 기준 분기 (gap=0이면 fresh)", () => {
    const slot = makeSlot({
      timeframe: "1W",
      freshAfterDays: 7,
      staleAfterDays: 21,
    });
    expect(getSlotFreshness(slot, "2026-06-02", "2026-06-02", [])).toBe(
      "fresh",
    );
  });

  it("1W 달력일 stale (gap > staleAfterDays)", () => {
    const slot = makeSlot({
      timeframe: "1W",
      freshAfterDays: 7,
      staleAfterDays: 21,
    });
    // asof 2026-05-01, latestPrice 2026-06-02 → 32일 gap.
    expect(getSlotFreshness(slot, "2026-05-01", "2026-06-02", [])).toBe(
      "stale",
    );
  });
});

describe("evaluateBandStaleness", () => {
  it("invalid date → not stale, gap 0", () => {
    expect(evaluateBandStaleness(null, "2026-06-02")).toEqual({
      isStale: false,
      gapBusinessDays: 0,
    });
    expect(evaluateBandStaleness("2026-06-02", null)).toEqual({
      isStale: false,
      gapBusinessDays: 0,
    });
  });
  it("gap ≤ threshold → not stale (gap만 반환)", () => {
    const r = evaluateBandStaleness("2026-06-02", "2026-06-02", 5);
    expect(r.isStale).toBe(false);
    expect(r.gapBusinessDays).toBeGreaterThanOrEqual(0);
  });
  it("gap > threshold → stale + reason 포함", () => {
    // bandAsof 2026-05-01, priceLatest 2026-06-02 → 영업일 gap > 5.
    const r = evaluateBandStaleness("2026-06-02", "2026-05-01", 5);
    expect(r.isStale).toBe(true);
    expect(r.gapBusinessDays).toBeGreaterThan(5);
    expect(r.reason).toContain("stale");
  });
});
