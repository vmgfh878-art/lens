// CP215 — 1D 화면 가격 lookback 상수.
// 밴드 응답의 마지막 asof_date 를 안전하게 덮을 만큼 넉넉히 받는다.
// fetchPrices 가 start/end 캘린더 범위를 받으므로 단위는 calendar days.
//
// PRICE_LOOKBACK_LIMIT_1D = 500   → 약 250 거래일 (~1.5y)
// PRICE_LOOKBACK_LIMIT_1W = 1825  → 약 260 주봉 (~5y)
export const PRICE_LOOKBACK_LIMIT_1D = 500;
export const PRICE_LOOKBACK_LIMIT_1W = 1825;

// CP215 — 가격 latest 와 밴드 latest asof_date 의 거래일 gap 이 이 값보다 크면
// 밴드를 stale 로 보고 토글 disable. 1D band slot.staleAfterDays(5) 와 일치.
export const BAND_STALE_THRESHOLD_BUSINESS_DAYS = 5;
