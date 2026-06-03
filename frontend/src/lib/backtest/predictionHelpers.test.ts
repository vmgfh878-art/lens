// CP230 characterization — backtest/predictionHelpers.ts 박제.

import { describe, expect, it } from "vitest";

import type { PredictionResult } from "@/api/client";

import {
  getBandWidthValue,
  getConservativeValue,
  getHighestUpperBandValue,
  getWorstLowerBandValue,
  median,
  percentileRank,
} from "./predictionHelpers";

function makePrediction(
  overrides: Partial<PredictionResult> = {},
): PredictionResult {
  return {
    conservative_series: [],
    line_series: [],
    lower_band_series: [],
    upper_band_series: [],
    forecast_dates: [],
    ...overrides,
  } as unknown as PredictionResult;
}

describe("median", () => {
  it("빈 배열 → null", () => {
    expect(median([])).toBeNull();
  });
  it("홀수 길이 → 가운데 값", () => {
    expect(median([1, 2, 3])).toBe(2);
    expect(median([3, 1, 2])).toBe(2);
  });
  it("짝수 길이 → 가운데 두 값 평균", () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });
  it("단일 원소", () => {
    expect(median([42])).toBe(42);
  });
});

describe("percentileRank", () => {
  it("빈 배열 → null", () => {
    expect(percentileRank(1, [])).toBeNull();
  });
  it("finite 만 필터링 후 ≤ 비율", () => {
    expect(percentileRank(2, [1, 2, 3, Number.NaN, Infinity])).toBe(2 / 3);
    expect(percentileRank(0, [1, 2, 3])).toBe(0);
    expect(percentileRank(3, [1, 2, 3])).toBe(1);
  });
});

describe("getWorstLowerBandValue", () => {
  it("finite 만 필터 후 Math.min", () => {
    const result = getWorstLowerBandValue(
      makePrediction({
        lower_band_series: [10, 5, Number.NaN, 8, Infinity, -1],
      }),
    );
    expect(result).toBe(-1);
  });
  it("전부 비finite 또는 빈배열 → null", () => {
    expect(
      getWorstLowerBandValue(makePrediction({ lower_band_series: [] })),
    ).toBeNull();
    expect(
      getWorstLowerBandValue(
        makePrediction({ lower_band_series: [Number.NaN, Infinity] }),
      ),
    ).toBeNull();
  });
});

describe("getHighestUpperBandValue", () => {
  it("finite 만 필터 후 Math.max", () => {
    expect(
      getHighestUpperBandValue(
        makePrediction({ upper_band_series: [1, 2, 3, Number.NaN, 7, 5] }),
      ),
    ).toBe(7);
  });
  it("빈배열 → null", () => {
    expect(
      getHighestUpperBandValue(makePrediction({ upper_band_series: [] })),
    ).toBeNull();
  });
});

describe("getBandWidthValue", () => {
  it("upper - lower", () => {
    expect(
      getBandWidthValue(
        makePrediction({
          lower_band_series: [10, 5],
          upper_band_series: [12, 14],
        }),
      ),
    ).toBe(14 - 5);
  });
  it("한쪽 null이면 null", () => {
    expect(
      getBandWidthValue(
        makePrediction({
          lower_band_series: [],
          upper_band_series: [10],
        }),
      ),
    ).toBeNull();
  });
});

describe("getConservativeValue", () => {
  it("conservative_series가 있으면 거기서 last finite", () => {
    expect(
      getConservativeValue(
        makePrediction({
          conservative_series: [1, 2, Number.NaN],
          line_series: [99, 100],
        }),
      ),
    ).toBe(2);
  });
  it("conservative_series가 비면 line_series로 폴백", () => {
    expect(
      getConservativeValue(
        makePrediction({
          conservative_series: [],
          line_series: [10, 20, 30],
        }),
      ),
    ).toBe(30);
  });
  it("둘 다 finite 없으면 null", () => {
    expect(
      getConservativeValue(
        makePrediction({
          conservative_series: [Number.NaN],
          line_series: [Number.NaN],
        }),
      ),
    ).toBeNull();
  });
});
