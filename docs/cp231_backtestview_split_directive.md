# CP231 BacktestView.tsx 분리 (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고, 검증하고, 차단 판단을 한다. 추측 금지. 막히면 멈추고 보고.

## 역할 고정
- **모드**: `code` (구현 모드). 지시받은 분리 작업을 직접 수행하고 같은 턴에 자가 점검만 보고한다.
- **권한**: 코드 수정, 로컬 검증(tsc / 단위테스트 / 스크린샷 diff)만.
- **금지**: 새 모델 학습, 새 calibration, DB write, Supabase 호출, 사용자가 직접 수정한 파일을 revert. 동작(수치·렌더 결과) 변경 금지 — 이 CP는 **순수 구조 이동**이다.
- **자가 점검**: 매 Step 종료 시 [Plan v3 정합] [구조 결함] [모델 영향] 3축 보고. (이 CP는 FE 전용이라 모델 영향은 거의 항상 N/A지만 그래도 명시.)
- **커밋 메시지**: 간결하게. 한 줄 요약 + 변경 요지. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 환경
- **워킹 디렉토리**: `C:\Users\user\lens` (프론트는 `C:\Users\user\lens\frontend`).
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128). 이 CP는 FE 전용이라 venv는 쓰지 않는다.
- **백엔드 기동**(스크린샷 검증에 필요): `scripts\start_demo.ps1` 또는 `uvicorn`. 프론트가 `NEXT_PUBLIC_BACKEND_URL` 로 백엔드를 호출하므로, 스크린샷 baseline 과 동일한 백엔드가 떠 있어야 시각 비교가 의미 있다.
- **프론트 기동**: `frontend` 에서 `npm run dev` (Next.js 14.2.3, 기본 포트 3000).
- **포트 충돌 회피**: 3000 이 이미 점유돼 있으면 Next 가 3001 로 올라간다. CP230 baseline 을 찍은 포트와 **같은 포트**에서 비교 스크린샷을 찍어라. 포트가 다르면 baseline 을 다시 찍지 말고, 같은 포트로 dev 서버를 재기동해서 맞춰라.
- **타입체크 명령**: `frontend` 에서 `npx tsc --noEmit` (이 repo 의 `tsconfig.json` 사용).

## 진단 (근거)
조사 출처: `frontend\src\components\BacktestView.tsx` 직접 Read (전체 1644줄), `frontend\src\lib\backtest\` 디렉토리 Glob/Read, `frontend\package.json` Read.

**문제 1 — 단일 파일 비대.** `frontend\src\components\BacktestView.tsx` 는 **1644줄**(마지막 줄 포함, 코드 본문 1643줄)이다. 이 안에 (a) 순수 시뮬레이션/통계 계산, (b) 신호 빌더, (c) 데이터 로딩, (d) SVG 차트 컴포넌트, (e) 거대한 React 렌더 트리가 한 파일에 섞여 있다. `BacktestView()` 컴포넌트 본체만 838~1643줄(약 805줄)이고, 그 위 87~836줄(약 749줄)이 전부 컴포넌트 밖 헬퍼다.

**문제 2 — 순수 함수가 컴포넌트 파일에 갇혀 테스트 불가.** 다음은 부작용 없는 순수 함수인데 컴포넌트 모듈 안에 있어 단위 테스트로 고정할 수 없다 (현재 줄 번호):

- 시뮬레이션/통계 (총 약 200줄):
  - `runStrategyBacktest` — **596~742줄 (147줄)**. 백테스트 핵심. `BacktestSimulationResult | null` 반환.
  - `calculateMaxDrawdown` — **543~553줄**
  - `calculateSharpe` — **555~563줄**
  - `calculateSortino` — **565~577줄**
  - `quantile` — **579~586줄**
  - `chooseLargeLossThreshold` — **588~594줄**
- 신호 빌더 (총 약 295줄):
  - `buildRawSignals` — **107~176줄**
  - `getRawTarget` — **178~209줄**
  - `normalizeRsi` — **211~217줄**
  - `buildIndicatorRows` — **219~250줄**
  - `getIndicatorRawTarget` — **252~287줄**
  - `buildIndicatorSignals` — **289~348줄**
  - `buildSignals` — **350~401줄**
- 카드 분류 헬퍼 (loadStrategySignalCard 가 사용):
  - `getLatestIndicatorBefore` — **403~413줄**
  - `getBandWidthState` — **415~426줄** (주의: `loadStrategySignalCard` 와 **JSX 렌더(1308, 1323줄)** 양쪽에서 쓴다 → export 필수)
  - `classifySignalGroup` — **428~446줄**
  - `classifyIndicatorSignal` — **448~462줄**
- SVG 차트 (총 약 93줄):
  - `buildPath` — **744~746줄**
  - `MiniLineChart` — **748~825줄**
  - `PositionStrip` — **827~836줄**
- 데이터 로딩 (선택 추출):
  - `extractErrorMessage` — **87~95줄**
  - `fetchPriceHistory` — **97~105줄**
  - `loadStrategySignalCard` — **464~541줄**

**문제 3 — 입출력 타입은 이미 분리돼 있다.** `frontend\src\lib\backtest\types.ts` (137줄)에 `BacktestSimulationResult`(66~91줄), `RiskSignal`(34~49줄), `BacktestPoint`(51~57줄), `TradeEvent`(59~64줄), `LineSeries`(93~97줄), `StrategySignalCard`(117~136줄) 등이 이미 정의돼 있고 BacktestView 가 51~63줄에서 type-only import 한다. 즉 **함수 본체만 옮기면 되고 타입은 새로 만들 필요가 없다.** 이게 이 분리를 저위험으로 만든다.

**기대 효과.** 위 (a)(b)(d)와 선택적 (e) 로딩을 `lib/backtest/*` + `components/backtest/*` 로 옮기면 `BacktestView.tsx` 는 상태/렌더 중심 **~900줄**로 줄고, 옮긴 순수 함수는 단위 테스트로 수치를 고정할 수 있다.

## 선행 의존
- **CP230 그린이 필수.** CP230 은 이 컴포넌트의 (1) 스크린샷 baseline 과 (2) 순수 함수 단위 테스트 환경을 구축하는 characterization CP다.
- **확인 사실(중요):** 현재 `frontend\package.json` 에는 **Vitest 도 Playwright 도 없다**(`devDependencies` 에 typescript / @types / tailwind / postcss / autoprefixer 만 존재). 따라서 단위 테스트 러너와 스크린샷 도구는 **CP230 이 설치·설정**하는 것이 전제다.
  - 이 지시서를 실행하기 전, 다음을 확인하라:
    1. `frontend\package.json` 에 테스트 러너 스크립트(예: `"test": "vitest run"` 또는 `vitest`)와 스크린샷 비교 절차가 존재한다.
    2. CP230 이 만든 BacktestView 스크린샷 **baseline** 이 존재한다(예: `frontend\tests\__screenshots__\` 또는 CP230 보고서에 명시된 경로).
    3. CP230 이 만든 순수 함수 단위 테스트가 **그린**이다.
  - **위 1~3 중 하나라도 없으면 즉시 중단하고 보고하라.** "테스트 환경이 없으니 그냥 추출만 하고 넘어간다"는 **금지**. 안전망 없는 구조 분리는 이 런북의 전역 규칙 위반이다.
  - 단, CP230 보고서가 "Vitest 대신 다른 러너/명령을 쓴다" 또는 "스크린샷 대신 다른 시각 회귀 절차를 쓴다"고 명시했다면 **그 절차를 따른다.** 이 지시서의 `vitest` / `playwright` 표기는 CP230 이 정한 실제 명령으로 치환해서 읽어라.

## 범위
**포함**
- `BacktestView.tsx` 내 순수 함수 4묶음을 새 모듈로 추출하고 caller(=BacktestView)를 새 모듈 import 로 이전한다.
  1. `lib/backtest/simulationEngine.ts` — 시뮬레이션 + 통계 순수 함수.
  2. `lib/backtest/signalBuilder.ts` — 신호 빌더 + 카드 분류 헬퍼.
  3. `components/backtest/Charts.tsx` — `MiniLineChart` + `PositionStrip` + `buildPath`.
  4. (선택) `lib/backtest/signalCardLoader.ts` — `loadStrategySignalCard` + `fetchPriceHistory` (+ `extractErrorMessage`).
- 각 추출 후 import 정리.

**제외**
- 동작/수치/렌더 결과 변경. (리팩토링 전용. 기능 변경 커밋과 분리.)
- 함수 signature, `BacktestSimulationResult` 등 타입, `BacktestView` 의 props/state 구조 변경.
- `lib/backtest/types.ts`, `constants.ts`, `evaluators.ts`, `predictionHelpers.ts` 의 내용 변경(이미 분리돼 있음 — import 만 한다).
- Supabase 관련 일체(보류 대상).
- 백엔드 코드, API 응답 schema.
- 사용자가 직접 수정한 파일의 revert.
- `loadStrategySignalCard` 안의 `/predictions/history` 관련 legacy TODO 주석(492, 982줄) 해결 — 이건 기능 변경이라 별도 CP. **주석째로 그대로 옮긴다.**

## Sub-step (Strangler Fig, 작은 단위)
원칙: **순수 함수 먼저(부작용 없는 계산) → I/O 경계 → 컴포넌트**. 각 Step 은 "새 모듈에 코드 추가(공존) → caller 를 새 모듈로 이전 → 옛 정의 제거"의 한 묶음이고, **한 Step = 한 커밋 = 한 revert 단위**다. 각 Step 끝에서 `npx tsc --noEmit` + 단위 테스트 + 스크린샷 diff 를 돌리고 그린이어야 다음 Step 으로 간다.

추출 함수가 의존하는 외부 import 는 새 모듈 상단에 그대로 가져가라. 정확한 매핑:
- `@/api/client` → `PriceBar`, `IndicatorPoint`, `PredictionResult`, (loader 는 추가로 `DisplayTimeframe`, `StockSummary`, `fetchPrices`, `fetchIndicators`, `fetchTickers`)
- `@/lib/backtest/types` → `StrategyId`, `BacktestSimulationResult`, `BacktestPoint`, `TradeEvent`, `RiskSignal`, `StrategySignalCard`, `LineSeries`
- `@/lib/backtest/constants` → `LENS_BALANCE_RULE`, `INDICATOR_BASELINE_RULE`, (loader 는 `DEFAULT_PRICE_LOOKBACK_DAYS`)
- `@/lib/backtest/predictionHelpers` → `getConservativeValue`, `getWorstLowerBandValue`, `getBandWidthValue`, `median`, `percentileRank`
- `@/lib/dateUtils` → (loader 는 `buildDefaultPriceWindow`, `sortPriceRows`, `sortUniqueByDate`)

### Step 1 — `simulationEngine.ts` 추출 (순수 계산 먼저)
대상: `calculateMaxDrawdown`(543~553), `calculateSharpe`(555~563), `calculateSortino`(565~577), `quantile`(579~586), `chooseLargeLossThreshold`(588~594), `runStrategyBacktest`(596~742).
- 의존성 주의: `runStrategyBacktest` 는 `buildSignals` 와 `buildIndicatorSignals` 를 호출한다(604~607줄). 이 둘은 Step 2 에서 옮긴다. **Step 1 시점에는** signalBuilder 가 아직 BacktestView 안에 있으므로, 두 방법 중 하나를 택한다:
  - (권장) Step 2 의 signalBuilder 추출을 **Step 1 보다 먼저** 해도 된다 — 의존 방향이 simulationEngine → signalBuilder 이므로 signalBuilder 를 먼저 독립시키면 깔끔하다. 실행자가 순서를 바꿔도 좋으나, **그 경우 아래 Step 2 를 Step 1 로 번호만 바꿔 동일 절차로 수행**하고 보고서에 순서 변경을 적어라.
  - (대안) Step 1 을 먼저 하려면, `runStrategyBacktest` 가 쓰는 `buildSignals`/`buildIndicatorSignals` 를 새 모듈에서 `@/components/BacktestView` 가 아니라 **곧 만들 `@/lib/backtest/signalBuilder`** 에서 import 하도록 적고, Step 2 에서 signalBuilder 를 채운다. 단 이러면 Step 1 종료 시 tsc 가 깨지므로 권장하지 않는다.
- 작업:
  1. `frontend\src\lib\backtest\simulationEngine.ts` 생성. 위 6개 함수를 그대로 옮긴다. `runStrategyBacktest` 와 `BacktestSimulationResult` 를 쓰는 쪽이 있으니 **`runStrategyBacktest` 는 `export`**. 나머지 통계 함수는 단위 테스트에서 직접 검증하려면 `export` 하라(테스트가 CP230 설계대로 통계 함수를 직접 부른다면 필수).
  2. BacktestView 에서 옛 정의 6개 삭제하고 `import { runStrategyBacktest } from "@/lib/backtest/simulationEngine";` 추가. (1001줄의 `runStrategyBacktest(...)` 호출이 새 import 를 가리키게.)
- 검증: `npx tsc --noEmit` → 0 에러. 단위 테스트(simulationEngine 대상) → 통과. 스크린샷 diff → 허용오차 내.
- 커밋: `refactor(fe): extract backtest simulation engine from BacktestView`.

### Step 2 — `signalBuilder.ts` 추출
대상: `buildRawSignals`(107~176), `getRawTarget`(178~209), `normalizeRsi`(211~217), `buildIndicatorRows`(219~250), `getIndicatorRawTarget`(252~287), `buildIndicatorSignals`(289~348), `buildSignals`(350~401), `getLatestIndicatorBefore`(403~413), `getBandWidthState`(415~426), `classifySignalGroup`(428~446), `classifyIndicatorSignal`(448~462).
- export 정책:
  - `buildSignals`, `buildIndicatorSignals` → simulationEngine 이 import → **export 필수.**
  - `getBandWidthState` → **JSX(1308, 1323줄)와 loadStrategySignalCard 양쪽 사용 → export 필수**, BacktestView 로 다시 import.
  - `getLatestIndicatorBefore`, `classifySignalGroup`, `classifyIndicatorSignal` → `loadStrategySignalCard` 만 사용. Step 4(선택)에서 loader 까지 분리하면 거기서만 쓰이지만, 지금은 **export** 해서 BacktestView 의 `loadStrategySignalCard` 가 import 하게 한다.
  - `buildRawSignals`, `getRawTarget`, `buildIndicatorRows`, `getIndicatorRawTarget`, `normalizeRsi` → 내부 의존. 단위 테스트가 이들을 직접 부른다면 export, 아니면 모듈 내부 유지. **CP230 테스트가 부르는 함수는 반드시 export**(테스트 import 실패 시 즉시 보고).
- 작업:
  1. `frontend\src\lib\backtest\signalBuilder.ts` 생성, 위 함수들 이동.
  2. BacktestView 에서 옛 정의 삭제. `import { buildSignals, buildIndicatorSignals, getBandWidthState, getLatestIndicatorBefore, classifySignalGroup, classifyIndicatorSignal, normalizeRsi } from "@/lib/backtest/signalBuilder";` 형태로 **실제 사용되는 것만** 추가. (`normalizeRsi` 는 534줄 `loadStrategySignalCard` 안에서 쓰임 → loader 를 분리하지 않는 한 BacktestView 가 import.)
  3. Step 1 의 simulationEngine 이 `buildSignals`/`buildIndicatorSignals` 를 `@/lib/backtest/signalBuilder` 에서 import 하도록 정리.
- 검증: `npx tsc --noEmit` → 0. 단위 테스트(signalBuilder + simulationEngine) → 통과. 스크린샷 diff → 허용오차 내.
- 커밋: `refactor(fe): extract signal builder from BacktestView`.

### Step 3 — `components/backtest/Charts.tsx` 추출
대상: `buildPath`(744~746), `MiniLineChart`(748~825), `PositionStrip`(827~836).
- 이 셋은 한 묶음(`MiniLineChart` 가 `buildPath` 사용). `frontend\src\components\backtest\` 디렉토리가 없으면 새로 만든다.
- 작업:
  1. `frontend\src\components\backtest\Charts.tsx` 생성. 파일 상단에 `"use client";` 불필요(부모가 이미 client, 순수 표현 컴포넌트). import: `import type { LineSeries, TradeEvent, BacktestPoint } from "@/lib/backtest/types";`. `buildPath` 는 모듈 내부 유지(혹은 export 불필요).
  2. `export function MiniLineChart(...)`, `export function PositionStrip(...)`.
  3. BacktestView 에서 옛 정의 3개 삭제, `import { MiniLineChart, PositionStrip } from "@/components/backtest/Charts";` 추가. JSX 사용처(1523, 1531, 1573줄)는 import 만 바뀌면 그대로 동작.
- 검증: `npx tsc --noEmit` → 0. 스크린샷 diff → **특히 이 Step 에서 시각 회귀 위험 최고** → 허용오차 내 필수. 단위 테스트 → 통과(차트는 단위 테스트 대상 아님, 회귀만 없으면 됨).
- 커밋: `refactor(fe): extract backtest chart components`.

### Step 4 (선택) — `signalCardLoader.ts` 추출
대상: `fetchPriceHistory`(97~105), `loadStrategySignalCard`(464~541). (`extractErrorMessage`(87~95)는 현재 **어디서도 호출되지 않는 dead code** 로 보임 — Grep 으로 사용처 0 확인 후, 사용처가 없으면 **이 Step 에서 삭제하지 말고** 그대로 두거나, 별도 "dead code 제거" 판단은 차단 트리거 아래 규칙대로 보고만 하고 넘어가라. 함부로 지우지 않는다.)
- 조건: Step 1~3 으로 이미 ~900줄 목표에 도달했으면 **이 Step 은 생략 가능**. 목표 미달이거나 loader 응집도를 높이고 싶을 때만 수행.
- 작업:
  1. `frontend\src\lib\backtest\signalCardLoader.ts` 생성. `fetchPriceHistory`, `loadStrategySignalCard` 이동. 이 모듈은 signalBuilder(`buildSignals`, `buildIndicatorSignals`, `getLatestIndicatorBefore`, `classifySignalGroup`, `classifyIndicatorSignal`, `normalizeRsi`)와 `@/api/client`, `@/lib/backtest/constants`(`DEFAULT_PRICE_LOOKBACK_DAYS`), `@/lib/dateUtils`, `@/lib/backtest/predictionHelpers` 를 import.
  2. BacktestView 는 `import { loadStrategySignalCard } from "@/lib/backtest/signalCardLoader";` 만 남기고, 더 이상 signalBuilder 의 카드 분류 헬퍼를 직접 import 하지 않아도 되면 그 import 를 정리(`getBandWidthState` 는 JSX 가 여전히 쓰므로 **유지**).
- 검증: `npx tsc --noEmit` → 0. 단위 테스트 → 통과. 스크린샷 diff → 허용오차 내(특히 signal board 카드 렌더, 초기 자동 로드 AAPL 결과 확인).
- 커밋: `refactor(fe): extract strategy signal card loader`.

### Step 5 — 최종 import 정리 + 줄 수 확인
- BacktestView 상단 import 블록에서 미사용 import 제거(tsc 의 `noUnusedLocals`/lint 가 잡으면 그에 따른다). 남아야 하는 import 는 렌더에 실제 쓰이는 것들(`MetricCard`, `StatusInline`, formatters, constants 의 `STRATEGIES`/`getStrategyDefinition`/`DEFAULT_FEE_BPS`/`SIGNAL_GROUPS`/... , evaluators, `getBandWidthState` 등).
- 줄 수 측정 후 성공 기준 표와 대조.
- 커밋(변경 있을 때만): `refactor(fe): tidy BacktestView imports after split`.

## 인터페이스 보존
- **함수 signature 불변.** `runStrategyBacktest(params)` 의 params/반환 타입(`BacktestSimulationResult | null`), `buildSignals(params)`, `buildIndicatorSignals(params)`, `MiniLineChart(props)`, `PositionStrip(props)`, `loadStrategySignalCard(ticker, strategyId)` 모두 시그니처를 바꾸지 않는다. 옮기기만 한다.
- **타입 불변.** `lib/backtest/types.ts` 의 export 타입은 손대지 않는다.
- **컴포넌트 계약 불변.** `BacktestView` 는 props 가 없고(`export default function BacktestView()`), state hook 구성(839~861줄)을 바꾸지 않는다. JSX 출력 동일.
- 만약 추출 과정에서 어떤 함수의 signature 를 바꿔야만 컴파일이 되는 상황이 오면(예: 순환 import) → **그 자리에서 멈추고**, 호출자 영향(누가 부르는지, 줄 번호)을 정리해서 보고하라. 임의 변경 금지.

## 성공 기준 (측정 가능)
| 항목 | 시작 | 목표 | 판정 |
| --- | --- | --- | --- |
| `BacktestView.tsx` 줄 수 | 1643 | **~900 (≤ 950)** | `(Get-Content frontend\src\components\BacktestView.tsx \| Measure-Object -Line).Lines` |
| 신규 모듈 | 0 | `simulationEngine.ts`, `signalBuilder.ts`, `Charts.tsx` (+ 선택 `signalCardLoader.ts`) 생성 | 파일 존재 |
| `npx tsc --noEmit` 에러 | 0 | **0 (신규 0)** | 명령 출력 |
| 단위 테스트(CP230 + 신규 lib) | 그린 | **전부 통과, 회귀 0** | 테스트 러너 출력 |
| 스크린샷 diff | baseline | **허용오차 내(시각 회귀 0)** | 스크린샷 비교 출력 |
| 백테스트 수치(AAPL 등) | baseline | **단위 테스트로 불변 확인** | 테스트 통과 |
| 예상 시간 | — | **2~3시간** | — |

## 검증
각 Step 후, `frontend` 디렉토리에서 (PowerShell):
```powershell
# 1) 타입체크 (필수, 모든 Step)
npx tsc --noEmit

# 2) 단위 테스트 — CP230 이 정한 실제 명령으로 치환. 예시:
npm test            # 또는 npx vitest run

# 3) BacktestView 줄 수 (Step 끝/최종)
(Get-Content src\components\BacktestView.tsx | Measure-Object -Line).Lines
```
스크린샷(시각 회귀)은 CP230 이 정한 절차를 따른다. 절차 예시(CP230 이 Playwright 를 깔았을 경우):
```powershell
# 백엔드 + 프론트 기동 후, baseline 과 동일 포트/뷰포트로:
npx playwright test   # 또는 CP230 이 정의한 스크린샷 비교 스크립트
```
기대 결과: tsc 0 에러 / 단위 테스트 전부 통과(신규 lib 테스트 포함) / 스크린샷 diff 허용오차 내 / BacktestView ≤ 950줄.

수동 눈 확인(스크린샷 도구가 모호할 때 보조): `npm run dev` 후 `/backtest` 화면에서 (1) 초기 AAPL 자동 로드 결과, (2) 전략 셀렉트 변경, (3) 신호 카드 클릭 → 상세 백테스트, (4) 가격 차트의 매수/매도 마커와 누적수익 차트가 분리 전과 동일한지 본다.

## 차단 트리거 (중요)
**다음 상황이면 즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**
1. **선행 의존 미충족** — CP230 의 스크린샷 baseline / 단위 테스트 / 테스트 러너 스크립트 중 하나라도 없음. (안전망 없이 분리 시작 금지.)
2. **스크린샷 diff 가 허용오차 초과** = 시각 회귀 발생. 분리로 렌더가 바뀌었다는 뜻 → 멈추고 어느 Step/어느 화면이 깨졌는지 보고.
3. **단위 테스트로 백테스트 수치가 달라짐**(예: `strategyReturnPct`, `maxDrawdownPct`, `feeAdjustedSharpe`, `tradeCount` 변동) = 순수 함수 이동 중 로직이 바뀜 → 멈추고 보고.
4. **tsc 에러가 신규 발생** 하고 단순 import 경로 수정으로 해소되지 않음(예: 순환 import, 타입 불일치) → 멈추고 보고.
5. **export 누락으로 CP230 테스트의 import 가 깨짐**(테스트가 부르던 함수를 모듈 내부로 숨김) → 멈추고 어떤 함수가 필요한지 보고.
6. **백엔드/환경변수 누락으로 프론트가 기동 실패**하거나 `/backtest` 가 빈 화면 → 스크린샷 비교 자체가 무의미 → 멈추고 보고.
7. **dead code 발견**(`extractErrorMessage` 등 사용처 0) → 임의 삭제 금지. 보고만 하고 그대로 둔다.
8. **사용자가 직접 수정한 파일과 충돌** → revert 금지, 보고.

## ADR
완료 후 `docs/adr/0021-backtestview-split.md` 1장 작성(200~300단어). 기록할 것: BacktestView 1643줄을 simulationEngine / signalBuilder / Charts(+선택 signalCardLoader)로 가른 **결정과 경계 기준**(순수 계산 / 신호 빌더 / 표현 컴포넌트 / I/O 로딩의 4분할), 타입을 `types.ts` 에 이미 분리해 둔 것이 저위험 이동을 가능케 한 점, `getBandWidthState` 가 빌더이면서 렌더에서도 쓰여 export 한 예외, signalCardLoader 를 선택 단계로 둔 이유, 그리고 CP230 의 스크린샷+단위테스트 안전망에 의존했다는 점. (`docs/adr/` 디렉토리가 비어 있으면 새로 만든다.)

## 자가 점검 결과 양식
완료 보고에 아래를 채운다.
- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ____ (FE 구조 분리로 밴드 본체·fidelity·cost 로직 변경 없음 여부)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ____ (순환 import / 경계 누수 / export 누락 여부)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ____ (학습·calibration·예측 수치에 영향 없음 — 통상 N/A)

## 산출물
- 변경 파일:
  - `frontend/src/components/BacktestView.tsx` (축소)
  - `frontend/src/lib/backtest/simulationEngine.ts` (신규)
  - `frontend/src/lib/backtest/signalBuilder.ts` (신규)
  - `frontend/src/components/backtest/Charts.tsx` (신규)
  - `frontend/src/lib/backtest/signalCardLoader.ts` (신규, 선택)
  - `docs/adr/0021-backtestview-split.md` (신규)
- `docs/cp231_report.md` — 요구 / 한 일(Step별 커밋 해시) / 결정 / 후속(예: signalCardLoader 미수행 시 사유, legacy `/predictions/history` TODO 잔존, dead code `extractErrorMessage` 처리 결과). 필요한 만큼만.
