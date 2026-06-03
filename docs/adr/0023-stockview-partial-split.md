# ADR-0023: StockView 부분 분리 — 순수 헬퍼/표현 위젯만, 상태 결합 로직은 보류

Status: Accepted
Date: 2026-06-04
CP: CP233

## 결정

`frontend/src/components/StockView.tsx` (시작 1041줄)을 두 묶음으로만 분리하고 메인 로직은 본체에 남긴다.

- **`lib/stock/helpers.ts`** — 순수 헬퍼 (`getLastPrice`, `getChangePercent`, `getLastFinite`, `getPriceLookbackDays`, `buildAiState`) + 얇은 fetch (`fetchPriceHistory`) + `AiState` 인터페이스 + `FULL_PRICE_HISTORY_START_YEAR` 상수.
- **`components/stock/IndicatorOptions.tsx`** — 보조지표 체크박스 위젯 (`IndicatorOptionGroup` named export + `IndicatorOption` module-local).

추가 dead code 정리: `getLastPoint` (caller 0건 Grep 확정) + `IndicatorChartPoint`/`IndicatorChartSeries` named import 제거 + `fetchPrices`/`PRICE_LOOKBACK_LIMIT_*`/`buildDefaultPriceWindow`/`buildFullPriceWindows`/`sortPriceRows` (헬퍼로 옮겨 본문 미사용) 제거.

## 메인 로직을 분리하지 않은 이유

`loadStock` (12조건 AI 상태 분기), 토글 disable 사유 우선순위, 신선도 판정/파생 라벨 (≈260줄)은 다음과 같이 강하게 결합되어 있다:

- React state hook 12개 이상에 직접 의존 — 컴포넌트 본체에서만 자연스럽게 read.
- 조건 분기 결과가 JSX 조건부 렌더 분기와 1:1 — props로 빼면 양쪽 진실의 출처가 둘이 된다.
- 신선도 라벨/배지는 `evaluateBandStaleness`/`getSlotFreshness` 호출 결과를 컴포넌트 상태와 합쳐 만들어 부수 효과 없는 함수로 분리하면 인자 폭증.

이 묶음을 떼면 분리 자체가 새 결합을 만든다 (custom hook으로 빼는 길이 있으나 그게 진짜 안전한지는 별도 안전망/리뷰가 필요). CP233은 **저위험 추출만**으로 범위를 한정.

## 줄수 목표 미달 (924 vs ≤905)

`StockView.tsx` 1041 → 924줄 (-11%). 지시서 허용 상한 905 대비 19줄 초과. 메인 로직 분리 보류로 더 줄일 여지가 없다 — 후속 CP에서 `loadStock` 또는 신선도 파생 라벨을 hook으로 분리하는 결정이 필요. 시각 회귀 0 + 동작 무변경이므로 19줄 초과는 안전성 vs 줄수 트레이드오프의 안전 쪽 선택.

## 안전망

CP230의 Playwright 4 screenshot + Vitest 166 passed (CP233 신규 헬퍼 14개 박제 포함) 매 Step 그린. 시각 회귀 0.

## 후속 (별도 CP)

- `loadStock` orchestration을 custom hook (`useStockLoader`)로 빼서 ≤905 도달 + 재사용성. 위험: state 12개 + ref/effect 의존을 hook 인터페이스로 노출하는 설계가 필요.
- 신선도/파생 라벨을 별도 selector lib로. `evaluateBandStaleness` 결과 + state 조합 → label 변환 순수 함수화.
