/**
 * CP220 — 운영 3 모델 (CP210 라인 / CP153 1D 밴드 / CP178 1W 밴드) 의 사용 데이터 정적 매니페스트.
 *
 * 단일 진리: `docs/v1_operating_models_reproducibility.md` §1~3 (데이터 섹션).
 * v1 동안 바뀌지 않으므로 코드로 박는다. v2 mlops 도입 시 fetcher 로 교체.
 *
 * 전문가용 노출 — 약어/식별자 그대로 (예: "atr_ratio", "v3_adjusted_ohlc"). 풀어쓰지 말 것.
 */
import type { ProductSlotId } from "@/lib/productSlots";

export interface FeatureRow {
  /** CSV/parquet 컬럼명 그대로 (예: "atr_ratio"). */
  name: string;
  /** 한 줄 설명. 전문가용 — 약어 풀이 X, 의미만. */
  description: string;
  /** coverage (결측 제외 비율). 1.000 / 0.998 등. 미수록 시 생략. */
  coverage?: string;
  /** source parquet 이름 등. 보통 통일이라 block-level 에 있으면 생략. */
  source?: string;
}

export interface UsageDataBlock {
  /** feature pack 식별자 (예: "F4_stress_delta_plus_yield_curve"). */
  featurePack: string;
  /** feature set (예: "price_volatility_volume"). */
  featureSet: string;
  /** feature version (예: "v3_adjusted_ohlc"). */
  featureVersion: string;
  /** target 정의 (예: "safe_line_score (수익률 단위 score, 화면 환산)"). */
  target: string;
  /** 데이터 provider (예: "yfinance"). */
  provider: string;
  /** 유니버스 (예: "S&P 500"). */
  universe: string;
  /** indicator parquet path (예: "data/parquet/indicators_yfinance_1D_500.parquet"). */
  parquetPath: string;
  features: FeatureRow[];
  /** 전처리 항목 한 줄씩. */
  preprocessing: string[];
  /** source_data_hash. 1W 는 단일 진리에 미수록 → 생략. */
  dataHash?: string;
}

const BASE_PRICE_FEATURES: FeatureRow[] = [
  { name: "log_return", description: "로그 수익률" },
  { name: "open_ratio", description: "시가 / 종가 비율" },
  { name: "high_ratio", description: "고가 / 종가 비율" },
  { name: "low_ratio", description: "저가 / 종가 비율" },
  { name: "vol_change", description: "거래량 변화율" },
  { name: "ma_5_ratio", description: "5일선 비율" },
  { name: "ma_20_ratio", description: "20일선 비율" },
  { name: "ma_60_ratio", description: "60일선 비율" },
  { name: "rsi", description: "Relative Strength Index" },
  { name: "macd_ratio", description: "MACD 비율" },
  { name: "bb_position", description: "Bollinger Band 내 위치" },
];

const LINE_USAGE: UsageDataBlock = {
  featurePack: "F4_stress_delta_plus_yield_curve",
  featureSet: "price_volatility_volume",
  featureVersion: "v3_adjusted_ohlc",
  target: "safe_line_score (수익률 단위 score, 화면에서 기준 종가에 환산)",
  provider: "yfinance",
  universe: "S&P 500",
  parquetPath: "data/parquet/indicators_yfinance_1D_500.parquet",
  features: [
    { name: "atr_ratio", description: "ATR / 가격 비율", coverage: "1.000" },
    { name: "vix_change_5d", description: "VIX 5일 변화", coverage: "0.998" },
    { name: "credit_spread_change_20d", description: "신용 스프레드 20일 변화", coverage: "0.993" },
    { name: "ma200_pct_change_20d", description: "200일선 20일 변화율", coverage: "0.993" },
    { name: "yield_curve", description: "장단기 금리차", coverage: "1.000" },
    ...BASE_PRICE_FEATURES,
  ],
  preprocessing: [
    "calendar_aligned split (시간 기준 train/val/test)",
    "walk-forward 4-fold (W1~W4)",
    "per-ticker normalize",
    "결측 forward-fill 후 drop",
  ],
  dataHash: "11bf3a4831d54815",
};

const BAND_1D_USAGE: UsageDataBlock = {
  featurePack: "price_volatility_volume",
  featureSet: "price_volatility_volume",
  featureVersion: "v3_adjusted_ohlc",
  target: "raw_future_return",
  provider: "yfinance",
  universe: "S&P 500",
  parquetPath: "data/parquet/indicators_yfinance_1D_500.parquet",
  features: BASE_PRICE_FEATURES,
  preprocessing: [
    "Stage 5T true walk-forward 3-fold (fold 마다 fresh checkpoint)",
    "per-ticker normalize",
    "validation-only lower_focused calibration fit",
    "test 고정 적용 (conformal 보정)",
  ],
  dataHash: "90666b44cbfb8e5c",
};

const BAND_1W_USAGE: UsageDataBlock = {
  featurePack: "price_volatility_volume",
  featureSet: "price_volatility_volume",
  featureVersion: "v3_adjusted_ohlc",
  target: "raw_future_return (주봉)",
  provider: "yfinance",
  universe: "S&P 500",
  parquetPath: "data/parquet/indicators_yfinance_1W_500.parquet",
  features: [
    ...BASE_PRICE_FEATURES.map((f) => {
      if (f.name === "log_return") return { ...f, description: "로그 수익률 (주봉)" };
      if (f.name === "ma_5_ratio") return { ...f, description: "5주선 비율" };
      if (f.name === "ma_20_ratio") return { ...f, description: "20주선 비율" };
      if (f.name === "ma_60_ratio") return { ...f, description: "60주선 비율" };
      return f;
    }),
    { name: "vix_close", description: "VIX 종가 (1W parquet 포함)" },
  ],
  preprocessing: [
    "Stage 5 true walk-forward 3-fold (1D 동일 calendar 정렬)",
    "per-ticker normalize",
    "WFLOCK walk-forward lower calibration (fold 별 별도 fit)",
  ],
  // 1W source_data_hash 단일 진리 미수록 → 생략
};

export const USAGE_DATA: Record<string, UsageDataBlock> = {
  "line-1d": LINE_USAGE,
  "band-1d": BAND_1D_USAGE,
  "band-1w": BAND_1W_USAGE,
};

export function getUsageData(slotId: ProductSlotId | string | null | undefined): UsageDataBlock | null {
  if (!slotId) {
    return null;
  }
  return USAGE_DATA[slotId] ?? null;
}
