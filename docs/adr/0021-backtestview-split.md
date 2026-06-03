# ADR-0021: BacktestView 1547줄을 4모듈로 분할

Status: Accepted
Date: 2026-06-04
CP: CP231

## 결정

`frontend/src/components/BacktestView.tsx` (시작 1547줄)을 시각/상태/렌더만 남기고 비-JSX 코드를 4개 모듈로 가른다. 경계 기준은 "코드가 무엇에 의존하느냐":

- **`lib/backtest/simulationEngine.ts`** — 시뮬레이션 + 통계 순수 함수 (`runStrategyBacktest`, `calculateMaxDrawdown`, `calculateSharpe`, `calculateSortino`, `quantile`, `chooseLargeLossThreshold`). 가격·신호만 입력받아 `BacktestSimulationResult`를 반환.
- **`lib/backtest/signalBuilder.ts`** — 신호 빌더 + 카드 분류 헬퍼 (`buildRawSignals`, `getRawTarget`, `normalizeRsi`, `buildIndicatorRows`, `getIndicatorRawTarget`, `buildIndicatorSignals`, `buildSignals`, `getLatestIndicatorBefore`, `getBandWidthState`, `classifySignalGroup`, `classifyIndicatorSignal`).
- **`components/backtest/Charts.tsx`** — SVG 표현 컴포넌트 (`MiniLineChart`, `PositionStrip` + 모듈-내부 `buildPath`).
- **`lib/backtest/signalCardLoader.ts`** — I/O 경계 (`fetchPriceHistory`, `loadStrategySignalCard`). 가격/지표/예측을 fetch한 뒤 signalBuilder를 호출해 카드를 생성. 이로써 BacktestView는 view-only가 된다.

## 가능했던 이유 — 타입 선행 분리

`lib/backtest/types.ts`에 `BacktestSimulationResult`, `RiskSignal`, `BacktestPoint`, `TradeEvent`, `LineSeries`, `StrategySignalCard`가 이미 정의돼 있었다. 함수 본체만 옮기면 됐고 타입 호환은 자동. 저위험 이동의 핵심 전제.

## 안전망 — CP230 의존

매 Step 후 Playwright 4 screenshot (report/stocks/backtests/training) diff 0 + Vitest 통과를 차단 트리거로 사용. Step C에서 simulationEngine/signalBuilder 단위 테스트를 stub에서 실 테스트로 활성화해 45개 수치 baseline을 박제 → 이후 Step D~F의 우발적 로직 변경 가드.

## 예외 — getBandWidthState

신호 빌더 모듈에 속하지만 JSX (전략 신호 카드 dl)에서도 사용 → `signalBuilder.ts`에서 export하고 BacktestView가 다시 import. 빌더와 표현 사이 경계가 깨끗하지는 않으나 의도된 결합. signalCardLoader 단계에서 다른 분류 헬퍼는 로더 내부로 흡수했지만 `getBandWidthState`만은 JSX 의존 때문에 노출.

## 결과

BacktestView 1547 → 839줄 (-46%). 4개 신규 모듈 생성, 운영 동작 무변경 (screenshot diff 0). Vitest 107 → 152 (회귀 0, +45 신규 박제).

## 남은 항목 (별도 CP 대상)

- `extractErrorMessage` (BacktestView.tsx:91)는 호출자 0건 dead code. CP231 §차단 트리거 #7 ("임의 삭제 금지")에 따라 그대로 두고 보고만.
- `loadStrategySignalCard` 안의 `/predictions/history` legacy TODO 주석은 기능 변경이라 별도 CP.
