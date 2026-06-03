// CP230 characterization — lib/training/formatters.ts 박제.

import { describe, expect, it } from "vitest";

import {
  extractErrorMessage,
  formatKoreanDateTime,
  formatMetric,
  formatModelLabel,
  formatRoleLabel,
  formatSignedNumber,
  formatSignedPctPoint,
  formatStatusLabel,
  formatValue,
} from "./formatters";

describe("formatStatusLabel", () => {
  it("completed → '완료'", () => {
    expect(formatStatusLabel("completed")).toBe("완료");
  });
  it("failed_nan → '실패'", () => {
    expect(formatStatusLabel("failed_nan")).toBe("실패");
  });
  it("failed_quality_gate → '기준 미달'", () => {
    expect(formatStatusLabel("failed_quality_gate")).toBe("기준 미달");
  });
  it("미정의 status → 그대로 반환", () => {
    expect(formatStatusLabel("running")).toBe("running");
  });
  it("null/undefined → '-'", () => {
    expect(formatStatusLabel(null)).toBe("-");
    expect(formatStatusLabel(undefined)).toBe("-");
  });
});

describe("formatRoleLabel", () => {
  it("line 계열 → '보수적 기준선'", () => {
    expect(formatRoleLabel("line_model")).toBe("보수적 기준선");
    expect(formatRoleLabel("line_v2")).toBe("보수적 기준선");
    expect(formatRoleLabel("line")).toBe("보수적 기준선");
  });
  it("band 계열 → 'AI 밴드'", () => {
    expect(formatRoleLabel("band_model")).toBe("AI 밴드");
    expect(formatRoleLabel("band")).toBe("AI 밴드");
  });
  it("composite_model → '이전 조합 실험'", () => {
    expect(formatRoleLabel("composite_model")).toBe("이전 조합 실험");
  });
  it("미정의 role → 그대로", () => {
    expect(formatRoleLabel("unknown")).toBe("unknown");
  });
  it("null → '-'", () => {
    expect(formatRoleLabel(null)).toBe("-");
  });
});

describe("formatModelLabel", () => {
  it("patchtst → 'PatchTST'", () => {
    expect(formatModelLabel("patchtst")).toBe("PatchTST");
  });
  it("cnn_lstm → 'CNN-LSTM'", () => {
    expect(formatModelLabel("cnn_lstm")).toBe("CNN-LSTM");
  });
  it("line_band_composite → '결합 방식 실험'", () => {
    expect(formatModelLabel("line_band_composite")).toBe("결합 방식 실험");
  });
  it("미정의 → formatValue 위임 ('-' 또는 string)", () => {
    expect(formatModelLabel("tide")).toBe("tide");
    expect(formatModelLabel(null)).toBe("-");
  });
});

describe("formatMetric", () => {
  it("null → fallback '-'", () => {
    expect(formatMetric(null, "rate")).toBe("-");
  });
  it("rate → value*100 + '%' (digits=1)", () => {
    expect(formatMetric(0.5, "rate")).toMatch(/^50([.,]\d+)?%$/);
  });
  it("pct_point → value*100 + '%p'", () => {
    expect(formatMetric(0.5, "pct_point")).toMatch(/^50([.,]\d+)?%p$/);
  });
  it("기본 number 포맷", () => {
    expect(formatMetric(1.23456789, "number")).not.toBe("-");
  });
  it("NaN → fallback", () => {
    expect(formatMetric(Number.NaN, "rate")).toBe("-");
  });
});

describe("formatValue", () => {
  it("number → Intl 4자리", () => {
    expect(formatValue(1.23456789)).not.toBe("-");
  });
  it("NaN/Infinity → '-'", () => {
    expect(formatValue(Number.NaN)).toBe("-");
    expect(formatValue(Infinity)).toBe("-");
  });
  it("string → 그대로", () => {
    expect(formatValue("hello")).toBe("hello");
  });
  it("boolean → 'true'/'false'", () => {
    expect(formatValue(true)).toBe("true");
    expect(formatValue(false)).toBe("false");
  });
  it("기타 → '-'", () => {
    expect(formatValue(null)).toBe("-");
    expect(formatValue(undefined)).toBe("-");
    expect(formatValue({})).toBe("-");
  });
});

describe("formatSignedNumber", () => {
  it("null → '-'", () => {
    expect(formatSignedNumber(null)).toBe("-");
  });
  it("양수에 '+' 부호", () => {
    expect(formatSignedNumber(1)).toMatch(/^\+1$/);
  });
  it("음수는 그대로(-)", () => {
    expect(formatSignedNumber(-1)).toMatch(/^-1$/);
  });
  it("0은 부호 없이", () => {
    expect(formatSignedNumber(0)).toBe("0");
  });
});

describe("formatSignedPctPoint", () => {
  it("null → '-'", () => {
    expect(formatSignedPctPoint(null)).toBe("-");
  });
  it("양수에 '+' 부호 + '%p'", () => {
    expect(formatSignedPctPoint(5)).toMatch(/^\+5([.,]\d+)?%p$/);
  });
  it("음수 + '%p'", () => {
    expect(formatSignedPctPoint(-3)).toMatch(/^-3([.,]\d+)?%p$/);
  });
});

describe("extractErrorMessage", () => {
  it("Network Error → 백엔드 안내 문자열", () => {
    const result = extractErrorMessage(new Error("Network Error"), "fallback");
    expect(result).toContain("백엔드");
    expect(result).toContain("NEXT_PUBLIC_BACKEND_URL");
  });
  it("ECONNREFUSED 포함 → 백엔드 안내", () => {
    const result = extractErrorMessage(
      new Error("connect ECONNREFUSED 127.0.0.1:8000"),
      "fallback",
    );
    expect(result).toContain("백엔드");
  });
  it("일반 Error → error.message", () => {
    expect(extractErrorMessage(new Error("oops"), "fallback")).toBe("oops");
  });
  it("non-Error → fallback", () => {
    expect(extractErrorMessage("string error", "fallback")).toBe("fallback");
    expect(extractErrorMessage(undefined, "fallback")).toBe("fallback");
  });
});

describe("formatKoreanDateTime", () => {
  it("null/undefined → '-'", () => {
    expect(formatKoreanDateTime(null)).toBe("-");
    expect(formatKoreanDateTime(undefined)).toBe("-");
  });
  it("invalid date string → 원본 반환", () => {
    expect(formatKoreanDateTime("not-a-date")).toBe("not-a-date");
  });
  it("valid ISO → ko-KR Asia/Seoul + ' KST' 접미", () => {
    // 2026-06-02T00:00:00Z = KST 2026-06-02 09:00 (UTC+9).
    const result = formatKoreanDateTime("2026-06-02T00:00:00Z");
    expect(result).toMatch(/ KST$/);
    expect(result).toContain("2026");
    expect(result).toContain("09:00");
  });
});
