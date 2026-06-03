# CP231 보고서 — BacktestView 분리

**완료일**: 2026-06-04
**선행 의존**: CP230 (Playwright + Vitest baseline, 그린)
**커밋 범위**: `a69712a` → `d9b031b` → `91bae17` → `0498b12` → `09cc266` → `93d1895` → `e4a2863` (6 commit)

## 요구

`frontend/src/components/BacktestView.tsx` (시작 1547줄, 지시서 진단 1643줄에서 CP224b dead-code 제거 후 1547)을 4개 모듈로 분할. 동작 무변경. 매 Step tsc + Vitest + Playwright screenshot diff로 회귀 가드.

## 한 일 (Step별)

| Step | 내용 | 커밋 | 추출 모듈 |
|---|---|---|---|
| 0 | baseline 그린 확인 (tsc 0, Vitest 107, Playwright 4 diff 0) | — | — |
| A | signalBuilder.ts (11 함수, 지시서 Step 2를 의존 방향상 먼저) | `d9b031b` | `lib/backtest/signalBuilder.ts` |
| B | simulationEngine.ts (6 함수, 지시서 Step 1) | `91bae17` | `lib/backtest/simulationEngine.ts` |
| C | CP230 stub 활성화 + 45 신규 박제 테스트 | `0498b12` | (테스트 파일 2개 수정) |
| D | Charts.tsx (3 컴포넌트, 지시서 Step 3) | `09cc266` | `components/backtest/Charts.tsx` |
| E | signalCardLoader.ts (2 함수, 지시서 Step 4 — 선택을 수행) | `93d1895` | `lib/backtest/signalCardLoader.ts` |
| F | unused import 정리 (지시서 Step 5) | `e4a2863` | — |

## 결정

- **Step 순서 변경**: 지시서 권장에 따라 signalBuilder (Step 2)를 simulationEngine (Step 1)보다 먼저 추출. 의존 방향이 simulationEngine → signalBuilder 이므로 자연스러운 순서.
- **signalCardLoader 추출 (선택 Step)을 수행**: Step D 종료 시 958줄 (목표 ≤950 8줄 초과). 추출 후 868 → Step F 정리 후 839줄. 응집도와 view-only 컴포넌트 목표 양쪽 충족.
- **모든 함수 export**: CP230 테스트가 import해 박제할 수 있도록.
- **`getBandWidthState`만 signalBuilder + BacktestView 양쪽에서 사용**: JSX (전략 신호 카드 dl)가 직접 호출 → export 예외 유지.

## 핵심 컴포넌트 존재 체크리스트

| 항목 | 확인 |
|---|---|
| `lib/backtest/simulationEngine.ts` 생성 | OK |
| `lib/backtest/signalBuilder.ts` 생성 | OK |
| `components/backtest/Charts.tsx` 생성 | OK |
| `lib/backtest/signalCardLoader.ts` 생성 | OK |
| `runStrategyBacktest` signature 보존 (BacktestSimulationResult \| null) | OK |
| `buildSignals` / `buildIndicatorSignals` signature 보존 | OK |
| `MiniLineChart` / `PositionStrip` props 인터페이스 보존 | OK |
| `loadStrategySignalCard(ticker, strategyId)` signature 보존 | OK |
| `getBandWidthState` JSX (1308, 1323 위치) 호출 유지 | OK |
| BacktestView state hook 구성 / props 인터페이스 무변경 | OK |
| `lib/backtest/types.ts` 무수정 | OK |

## 새 테스트 결과

- `simulationEngine.test.ts`: 18 tests passed (calculateMaxDrawdown 4 / calculateSharpe 3 / calculateSortino 3 / quantile 2 / chooseLargeLossThreshold 2 / runStrategyBacktest 4).
- `signalBuilder.test.ts`: 27 tests passed (normalizeRsi 3 / getBandWidthState 4 / getRawTarget 4 / classifySignalGroup 4 / classifyIndicatorSignal 4 / getLatestIndicatorBefore 4 / buildSignals 2 / buildIndicatorSignals 2).
- 합계 45 신규. Vitest 107 → 152 passed. 박제값은 Step B 추출 직후 시점 결과로 → Step D, E, F에서 동일 결과 유지 확인 (회귀 0).
- 박제 실수 3건 발견·정정 (calculateSharpe mean 계산 실수, calculateSortino 동일, buildSignals reentryConfirmDays 트레이싱). 실제 함수 값에 맞춰 expected 수정.

## dry-run 결과 (Playwright screenshot diff)

매 Step 후 4 screenshot (report / stocks / backtests / training, chromium-win32) 비교. **모든 Step에서 diff 0** (`maxDiffPixelRatio` 0.01 임계 내). 시각 회귀 없음 = 분리 전후 렌더 동일.

| Step | tsc | Vitest | Playwright |
|---|---|---|---|
| 0 (baseline) | 0 | 107 passed \| 12 todo | 4 passed, diff 0 |
| A | 0 | 107 passed \| 12 todo | 4 passed, diff 0 |
| B | 0 | 107 passed \| 12 todo | 4 passed, diff 0 |
| C | 0 | 152 passed \| 4 todo | (테스트 추가만, 동작 무변경 → skip) |
| D (시각 회귀 위험 최고) | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |
| E | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |
| F (import 정리) | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |

## 기존 회귀 통과 건수

- `npx tsc --noEmit`: 0 에러 (전 Step). Baseline 0 → 최종 0.
- Vitest: baseline 107 passed → 최종 152 passed. 회귀 0 (기존 5 lib 테스트 전부 그대로 통과).
- Playwright: baseline 4 passed → 최종 4 passed, diff 0.

## 줄수 측정

| 시점 (commit) | BacktestView.tsx 줄수 |
|---|---|
| 시작 (a69712a) | 1547 |
| Step A 후 (d9b031b) | 1227 |
| Step B 후 (91bae17) | 1045 |
| Step C 후 (0498b12) | 1045 (테스트만 추가, BacktestView 무변경) |
| Step D 후 (09cc266) | 958 |
| Step E 후 (93d1895) | 868 |
| Step F 후 (e4a2863) | **839** |

목표 ≤950 충족. 시작 대비 -708줄 (-46%).

## 자가 점검 결과

- **[Plan v3 정합]** PASS — 사유: FE 구조 분리 only. 밴드 본체·fidelity·cost 로직 변경 0. AsymmetricHuberLoss α=1/β=2, RevIN, calendar future covariate 등 모델/학습 코드 무관.
- **[구조 결함]** PASS — 사유: 순환 import 없음 (simulationEngine → signalBuilder, signalCardLoader → signalBuilder, Charts → types 단방향). 모든 추출 함수 signature 무변경. CP230 안전망에 의존해 회귀 가드.
- **[모델 영향]** PASS (N/A 확정) — 사유: 학습·calibration·예측 수치 변경 0. backend/ai/ 코드 무수정. parquet/DB write 0. 백엔드 endpoint 응답 schema 무변경.

## 후속 (별도 CP)

- **`extractErrorMessage` (BacktestView.tsx:91) dead code**: 호출자 0건 (Grep 확정). 차단 트리거 #7 ("임의 삭제 금지")에 따라 보고만 하고 그대로 둠. 별도 청소 CP에서 처리.
- **`loadStrategySignalCard` 안의 `/predictions/history` legacy TODO** (signalCardLoader.ts): 기능 변경 (v1 endpoint로 이전)이라 별도 CP.
- **CP230 forward-stub 남은 4 todo**: CP232 (signalBuilder.test.ts 기존 4 todo는 CP231에서 새 테스트로 대체 — 활성화됨. 남은 4 todo는 다른 stub 파일).

## ADR

`docs/adr/0021-backtestview-split.md` 작성. 4분할 결정, 타입 선행 분리가 가능케 한 점, `getBandWidthState` export 예외, signalCardLoader 선택 단계 수행 사유, CP230 안전망 의존 기록.
