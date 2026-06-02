# CP230 프론트엔드 Characterization 안전망 (Directive)

> 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 CP 지시서.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측 금지, 막히면 멈추고 보고.

---

## 역할 고정

- **모드**: `code` (구현 + 같은 턴 자가 점검).
- **권한**: 코드 수정, 로컬 검증(lint / tsc / Vitest / Playwright 로컬 실행), 새 의존성(devDependency) 설치.
- **금지**:
  - 새 모델 학습 / 새 calibration 실행 금지.
  - DB write 금지, Supabase 호출·스키마 변경 금지.
  - 사용자가 직접 수정한 파일 revert 금지(직접수정 흔적 보이면 멈추고 보고).
  - **프로덕션 코드(`frontend/src/**`)의 동작 변경 금지.** 이 CP는 테스트 인프라와 baseline만 추가한다. 단 하나 예외는 Step 4의 결정론화(아래 "인터페이스 보존" 참조)로, 그조차 기존 호출자 동작을 바꾸지 않는 선에서만 한다.
- **자가 점검**(완료 시 양식대로 보고): Plan v3 정합 / 구조 결함 / 모델 영향.
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens` (프론트 작업은 `C:\Users\user\lens\frontend`).
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128) — 이 CP에서 Python은 백엔드 기동에만 쓴다.
- **프론트 런타임**: Next 14.2.3 / React 18 / TypeScript 5 / Node(로컬 설치본). `frontend/package.json` 기준 현재 devDependencies에 테스트 프레임워크 **없음**.
- **백엔드 기동**: `powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1`
  - 백엔드 `127.0.0.1:8000`, 프론트 `127.0.0.1:3000` 둘 다 기동한다.
  - 로컬 스냅샷 모드(`LENS_USE_LOCAL_SNAPSHOTS=1`, `data\parquet`)로 뜬다 → 데모 데이터 결정론에 유리.
  - 프론트는 `NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000` 으로 뜬다.
- **포트 충돌 주의**: `start_demo.ps1` 은 3000 점유 시 `node/cmd/npm` 프로세스만 종료한다. Playwright는 이미 떠 있는 `127.0.0.1:3000` 에 붙는 방식으로 구성하고(아래 Step 1), Playwright가 별도 dev 서버를 또 띄우지 않게 한다(이중 기동 금지).
- **tsc 확인**: `frontend` 에 별도 `lint` 스크립트 없음. 타입 체크는 `npx tsc --noEmit` 로 한다(`tsconfig.json`: `noEmit: true`, `target: es5`, `paths: @/* → ./src`).

---

## 진단 (근거)

프론트 분리 CP(CP231~234)는 **거대 뷰 컴포넌트에서 순수 로직을 추출**하는 작업이다. 현재 안전망이 0이라, 추출 중 회귀가 나도 잡을 장치가 없다. 직접 조사한 근거:

1. **거대 뷰 3종 — 추출 대상, 회귀 위험 집중처** (`wc -l` 직접 측정):
   - `frontend/src/components/BacktestView.tsx` — **1643줄**. 내부에 인라인 순수 로직 다수. 예: `buildSignals(params: {...})` (`BacktestView.tsx:350`). 이게 CP232 `signalBuilder` 추출 대상.
   - `frontend/src/components/TrainingView.tsx` — **1560줄**. 인라인 `buildExperimentCards(detail: AiRunDetail)` (`TrainingView.tsx:1104`). CP233 `experimentBuilder` 추출 대상.
   - `frontend/src/components/StockView.tsx` — **1097줄**. CP231 `simulationEngine` 및 overlay 로직이 여기와 BacktestView에 분산.
   - 이 세 파일은 한 줄만 잘못 옮겨도 화면이 깨지는데, 지금은 그걸 알려줄 테스트가 없다.

2. **이미 추출돼 있으나 테스트 없는 순수 함수들**(CP231~233 입출력 고정의 1차 대상):
   - `frontend/src/lib/backtest/evaluators.ts` — **62줄**, 함수 5개: `describeAvoidanceStrength`, `evaluateFollowReturn`, `evaluateDrawdown`, `evaluateLossAvoidance`, `evaluateTradeFrequency`. 전부 임계값 분기("강함/보통/약함" 등) → 분기 경계가 회귀 시 조용히 틀어진다.
   - `frontend/src/lib/backtest/predictionHelpers.ts` — **55줄**, 함수 6개: `getConservativeValue`, `getWorstLowerBandValue`, `getHighestUpperBandValue`, `getBandWidthValue`, `median`, `percentileRank`.
   - `frontend/src/lib/formatters.ts` — **94줄**, export 12개(`formatNumber`, `formatPercent`, `formatRatio`, `formatVolume`, `formatRatioAsPercent`, `formatCompact`, `normalizeRsi`, `finiteOrNull` 등). `Intl.NumberFormat("ko-KR")` 사용 → 로케일 의존.
   - `frontend/src/lib/training/formatters.ts` — **134줄**, export 12개+. 그중 `formatKoreanDateTime` (`training/formatters.ts:24`)은 `Asia/Seoul` 타임존 고정 `Intl.DateTimeFormat` → 입력 문자열만 고정하면 결정론.
   - `frontend/src/lib/staleness.ts` — **123줄**, `getSlotFreshness` / `evaluateBandStaleness` 등. 인자로 날짜를 받으므로(내부에서 `new Date()` 호출 안 함) 결정론적 테스트 가능.

3. **스크린샷 비결정성 원인**(직접 grep, `frontend/src/`):
   - `frontend/src/lib/dateUtils.ts:42` `const end = new Date();` 와 `dateUtils.ts:52` `new Date().getFullYear()` — `buildDefaultPriceWindow(lookbackDays)` 가 **현재 시각**으로 가격 윈도우를 만든다. 데모 뷰가 mount 시 이걸 호출하면 매일 다른 start/end → 백엔드 응답·차트가 매번 달라짐.
   - `frontend/src/lib/formatters.ts` 및 `training/formatters.ts` 의 `Intl` 로케일/타임존 — 실행 머신 TZ에 따라 흔들릴 수 있음(고정 필요).
   - `frontend/src/components/Chart.tsx:284,293` `requestAnimationFrame(updateForecastMarker)` — 캔버스 마커 비동기 갱신 → 스크린샷 타이밍 흔들림.
   - `frontend/src/components/BacktestView.tsx:1035,1045` `window.setTimeout(... scrollIntoView, 150)` — 스크롤 애니메이션.
   - 결론: **스크린샷은 결정론적 화면으로 좁혀야 한다.** `ReportView`(지표 가이드)는 아래 4번처럼 fetch 0, 시간 의존 0 → 1순위 baseline. 나머지 3화면은 "로드 성공(smoke)"을 1차 보증으로 하고, 스크린샷은 마스킹/대기 안정화 후 baseline을 뜬다.

4. **4화면 진입 구조**(`frontend/src/components/DashboardPage.tsx:11-39` 직접 확인):
   - 단일 페이지 `/` 에서 `?view=` 쿼리로 전환: `stocks` / `backtests` / `training` / `report`.
   - `stocks` → `StockView`(주식 보기), `backtests` → `BacktestView`(백테스트), `training` → `TrainingView`(AI 모델), `report` → `ReportView`(지표 가이드).
   - `ReportView` (`frontend/src/components/ReportView.tsx`, **125줄**)는 import에 API 클라이언트 없음·`useEffect`/fetch 없음·정적 JSX만 → **백엔드 불필요, 완전 결정론**.
   - `StockView`/`BacktestView`는 mount 시 기본 티커 `"AAPL"` 로 백엔드 호출(`StockView.tsx:205-206`, `BacktestView.tsx:839-840`). → 백엔드 기동 전제 또는 네트워크 모킹 필요.

5. **백엔드 의존 경로**(`frontend/src/api/baseClient.ts:8,68-77`): 로컬 호스트(`localhost`/`127.0.0.1`)면 프록시 없이 `http://127.0.0.1:8000` 직결. → 로컬 Playwright는 `start_demo.ps1` 로 백엔드를 띄워두면 그대로 붙는다.

**조사 출처**: 위 모든 줄번호/함수명/줄수는 `frontend/` 트리에 대한 직접 Read/Grep/`wc -l` 결과(2026-06-02 기준).

---

## 선행 의존

- **없음**(이 CP가 프론트 분리의 선행 안전망이다). CP230이 그린이어야 CP231~234(프론트 구조 분리)를 시작할 수 있다.
- 단, 검증 실행에는 로컬 백엔드가 떠야 하므로 `scripts\start_demo.ps1` 가 정상 기동 가능한 상태여야 한다(데이터 스냅샷 `data\parquet` 존재). 기동 자체가 안 되면 차단 트리거.

---

## 범위

### 포함
- `frontend` 에 **Playwright** 설치 + 설정 + 4화면 로드 smoke 테스트 + screenshot baseline 저장(git 커밋).
- `frontend` 에 **Vitest** 설치 + 설정 + 순수 함수 단위 테스트 작성.
  - 1차(현존 함수): `lib/backtest/evaluators.ts`, `lib/backtest/predictionHelpers.ts`, `lib/formatters.ts`, `lib/training/formatters.ts`, `lib/staleness.ts`.
  - 2차(전방 고정 stub): CP231 `simulationEngine`, CP232 `signalBuilder`, CP233 `experimentBuilder` — **아직 미추출**이므로, 현재 인라인 로직(`BacktestView.tsx:350` `buildSignals`, `TrainingView.tsx:1104` `buildExperimentCards`)을 보고 **기대 입출력 케이스를 `.todo`/skip 형태로 미리 기록**해 둔다(아래 Step 4-c). 추출 후 CP231~233에서 skip 해제만 하면 회귀가 잡히도록.
- 결정론 안정화: 스크린샷 대상 화면의 동적 요소 마스킹 + 네트워크/애니메이션 정착 대기.

### 제외
- **프로덕션 동작 변경 금지**(구조 추출은 CP231~234에서). 이 CP는 코드를 옮기지 않는다.
- **Supabase 관련 일체 보류**(호출·테스트·모킹 전부 제외).
- 사용자 직접수정 파일 변경 금지.
- CI 파이프라인 연동(GitHub Actions 등) 제외 — 로컬 실행 가능까지만.
- 컴포넌트 단위(React) 렌더 테스트 제외(스코프는 "순수 함수" + "로드 smoke/스크린샷"). 거대 뷰 자체의 단위 테스트는 분리 후 CP에서.

---

## Sub-step (Strangler Fig, 작은 단위)

> 안전망 구축 CP라 "옛 코드 제거"는 없지만, **한 Step = 한 revert 단위 = 1 커밋**을 지킨다. 각 Step 끝에 명시된 검증을 통과해야 다음 Step으로 간다. 실패 시 그 Step에서 멈추고 보고.

### Step 1 — Playwright 설치 + config (기존 dev 서버에 붙는 모드)
1. `frontend` 에서 설치:
   ```powershell
   npm install -D @playwright/test
   npx playwright install chromium
   ```
   (브라우저는 chromium 1종만. 결정론·속도 목적.)
2. `frontend/playwright.config.ts` 생성. 핵심 설정:
   - `testDir: "./tests/e2e"`.
   - `use.baseURL: "http://127.0.0.1:3000"`.
   - `use.viewport: { width: 1280, height: 800 }` (고정).
   - `use.locale: "ko-KR"`, `use.timezoneId: "Asia/Seoul"` (스크린샷/포맷 결정론).
   - `expect.toHaveScreenshot`: `maxDiffPixelRatio: 0.01`, `animations: "disabled"`.
   - **`webServer` 블록은 두지 않는다**(이미 `start_demo.ps1` 로 떠 있는 서버에 붙음 → 포트 이중 기동 방지). 대신 README/주석으로 "선행: start_demo.ps1 기동" 명시.
   - `fullyParallel: false`, `workers: 1`, `retries: 0` (결정론·디버그 용이).
3. `frontend/package.json` scripts에 추가(기존 dev/build/start 유지, 덮어쓰지 말 것):
   - `"test:e2e": "playwright test"`
   - `"test:e2e:update": "playwright test --update-snapshots"`
4. `.gitignore`(frontend) 정리: `test-results/`, `playwright-report/`, `blob-report/`, `.playwright/` 는 ignore. **단 `tests/e2e/**/__screenshots__/` (baseline)는 커밋 대상이므로 ignore하지 말 것.**
- **검증**: `npx tsc --noEmit` 0 에러. `npx playwright --version` 출력. **커밋**: `test(fe): add playwright config (CP230 step1)`.

### Step 2 — 4화면 로드 smoke + screenshot baseline
1. 선행: `powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1` 로 백엔드+프론트 기동(둘 다 ready 확인).
2. `frontend/tests/e2e/screens.smoke.spec.ts` 작성. 4개 화면 각각:
   - 진입: `page.goto("/?view=report")` 등 4개(`report` / `stocks` / `backtests` / `training`).
   - **로드 smoke 단정**(화면별 안정 셀렉터로):
     - `report`: `ReportView` 의 `h1` "예측선은 방향을, 밴드는 흔들림을 봅니다." 가시(`ReportView.tsx:35`). fetch 없음 → `networkidle` 불필요.
     - `stocks` / `backtests` / `training`: 해당 뷰 헤더(`view-header` 등 고정 텍스트/eyebrow)가 가시 + 콘솔 `pageerror`/`console.error` 0건. 데이터 fetch는 `await page.waitForLoadState("networkidle")` 로 정착.
   - **에러 가드**: 각 테스트에서 `page.on("pageerror")` 와 `page.on("console", msg => msg.type()==="error")` 를 수집해 비어 있어야 통과. (화면이 흰 화면/예외면 즉시 fail → 차단 트리거로 연결.)
3. **screenshot baseline**:
   - `report` 화면: `await expect(page).toHaveScreenshot("report.png")` — fetch·시간 의존 0이라 가장 안정. **이게 1순위 baseline.**
   - 나머지 3화면: 스크린샷 전에 **동적 영역 마스킹**으로 비결정성 차단:
     - `toHaveScreenshot({ mask: [page.locator(<날짜/시각/freshness 배지>), page.locator("canvas")], animations: "disabled" })`.
     - 차트 캔버스(`Chart.tsx`)는 `requestAnimationFrame` 마커 때문에 통째 마스킹.
     - freshness/asof 텍스트(staleness 기반)도 마스킹(데모 데이터라도 표시 포맷이 TZ 영향 가능).
   - 최초 실행은 baseline 생성: `npm run test:e2e:update`.
4. 같은 명령을 **연속 2회** 돌려 스크린샷이 재현되는지 직접 확인(`npm run test:e2e` 2회 연속 그린). 한 번이라도 diff 나면 → 마스킹 보강 후 재시도, 그래도 안 잡히면 차단 트리거(비결정).
- **검증**: 4 smoke 통과 + `tests/e2e/**/__screenshots__/` 에 baseline png 생성 + 2회 연속 그린. **커밋**: `test(fe): add 4-screen smoke + screenshot baseline (CP230 step2)` (baseline png 포함).

### Step 3 — Vitest 설치 + config
1. `frontend` 에서 설치:
   ```powershell
   npm install -D vitest @vitest/coverage-v8
   ```
   (DOM 불필요 — 순수 함수만. `jsdom` 설치하지 않음. 환경 `node`.)
2. `frontend/vitest.config.ts` 생성:
   - `test.environment: "node"`, `test.include: ["src/**/*.test.ts"]`.
   - `test.globals: false`(명시 import 사용).
   - `resolve.alias`: `@` → `<frontend>/src` (tsconfig `paths` 와 일치, `import` 메타 기반 절대경로).
   - **주의**: Vitest가 Playwright 스펙(`tests/e2e/**`)을 집지 않도록 `include` 를 `src/**` 로 한정.
3. `frontend/package.json` scripts:
   - `"test:unit": "vitest run"`
   - `"test:unit:watch": "vitest"`
- **검증**: 빈 통과(아직 테스트 0개여도) `npx vitest run` 이 설정 에러 없이 종료 + `npx tsc --noEmit` 0. **커밋**: `test(fe): add vitest config (CP230 step3)`.

### Step 4 — 순수 함수 단위 테스트 작성
> 입출력을 **현재 동작 그대로** 고정한다(=characterization). "이상적 값"이 아니라 "지금 코드가 내는 값"을 박는다. 분기 경계값을 반드시 포함.

- **4-a. 현존 함수 테스트**(파일은 대상 옆에 `*.test.ts`):
  - `src/lib/backtest/evaluators.test.ts` — 5개 함수 × 경계 케이스:
    - `describeAvoidanceStrength`: `0.6→"강함"`, `0.59→"보통"`, `0.4→"보통"`, `0.39→"약함"`, `null→"-"`, `NaN→"-"` (`evaluators.ts:9-15`).
    - `evaluateFollowReturn`: `result=null→"-"`, `buyHoldReturnRatio=null→"-"`, 방어 우위 케이스(`buyHoldReturnPct<0 && strategyReturnPct>buyHoldReturnPct`), `ratio 0.7→"양호"`, `0.4→"보통"`, `0.39→"약함"` (`evaluators.ts:18-32`).
    - `evaluateDrawdown`: `maxDrawdownImprovementPct 5→"양호"`, `0.1→"보통"`, `0→"약함"` (`evaluators.ts:34-45`).
    - `evaluateLossAvoidance`: `describeAvoidanceStrength` 위임 확인(`evaluators.ts:47-49`).
    - `evaluateTradeFrequency`: `tradeCount 20→"적정"`, `40→"많음"`, `41→"과도"`, `null→"-"` (`evaluators.ts:51-62`).
  - `src/lib/backtest/predictionHelpers.test.ts` — `median([])→null`, `median([1,2,3])→2`, `median([1,2,3,4])→2.5`; `percentileRank(value, values)` finite 필터 동작; `getWorstLowerBandValue`/`getHighestUpperBandValue` 의 `Number.isFinite` 필터 + 빈배열 `null`; `getBandWidthValue` = upper-lower (`predictionHelpers.ts:23-55`). PredictionResult는 최소 mock 객체로.
  - `src/lib/formatters.test.ts` — `formatNumber(null)→"-"`, `formatPercent(5)→"+5.00%"`, `formatPercent(-3)→"-3.00%"`, `formatRatio(0.5)→"50.0%"`, `formatRatio(2)→"2.00"`, `normalizeRsi(0.5)→50`, `normalizeRsi(70)→70`, `finiteOrNull(NaN)→null`, `formatRatioAsPercent(0.05)→"+5.00%"`, `formatUnsignedRatioAsPercent(0.05)→"5.00%"` (`formatters.ts`). **주의**: `Intl` 결과가 환경 TZ/로케일 의존이면 Vitest config가 `ko-KR` 을 강제하지 못할 수 있으니, 테스트 상단에서 값 자체(부호·자릿수)만 단정하고, 천단위 구분자 등 로케일 표기 차이는 정규식/`toContain` 으로 완화. 만약 머신 로케일 때문에 결과가 흔들리면 보고(차단 후보).
  - `src/lib/training/formatters.test.ts` — `formatStatusLabel("completed")→"완료"`, `formatRoleLabel("band_model")→"AI 밴드"`, `formatModelLabel("patchtst")→"PatchTST"`, `formatMetric(0.5,"rate")→"50.0%"`, `formatMetric(0.5,"pct_point")→"50.0%p"`, `formatSignedNumber(1)→"+..."`, `extractErrorMessage(new Error("Network Error"), "x")→` 백엔드 연결 안내 문자열 포함; `formatKoreanDateTime("2026-06-02T00:00:00Z")` 는 `KST` 접미 + 고정 결과(TZ는 함수가 `Asia/Seoul` 고정이므로 머신 무관) (`training/formatters.ts`).
  - `src/lib/staleness.test.ts` — `getSlotFreshness`: `slot=null→"empty"`, `refreshPolicy "deferred"→"deferred"`, `"static"→"static"`, 1D 거래일 카운트 기반 fresh/delayed/stale 분기, 1W 달력일 분기; `evaluateBandStaleness(priceLatest, bandAsof, threshold)` gap > threshold 시 `isStale:true` + reason 문자열. **모든 날짜를 인자로 고정**(내부 `new Date()` 없음 — 결정론).
- **4-b. 비결정 함수 주의**: `src/lib/dateUtils.ts` 의 `buildDefaultPriceWindow`(`:41`)·`buildFullPriceWindows`(`:51`)는 `new Date()` 의존이라 **이 CP에서 테스트하지 않는다**(혹은 `vi.setSystemTime` 으로 고정해야만 테스트 — 선택). 테스트하려면 `vi.useFakeTimers()` + `vi.setSystemTime(new Date("2026-06-02"))` 로 고정한 케이스만. 시간 고정 없이 단정하면 매일 깨지므로 금지.
- **4-c. 전방 고정 stub**(CP231~233 추출 대상의 입출력 예약):
  - `src/lib/backtest/simulationEngine.test.ts`, `src/lib/backtest/signalBuilder.test.ts`, `src/lib/training/experimentBuilder.test.ts` 3파일 생성.
  - 현재 인라인 구현(`BacktestView.tsx:350` `buildSignals`, `TrainingView.tsx:1104` `buildExperimentCards`, 그리고 BacktestView 내 시뮬레이션 루프)을 읽고 **대표 입력→기대 출력 케이스를 주석/`it.todo()` 로 명문화**. 아직 모듈이 없으니 `describe.skip` 또는 `it.todo` 로 두고, CP231~233에서 추출·import 연결 시 skip만 풀면 회귀가 잡히게 한다.
  - 각 stub 파일 상단에 `// CP231/232/233에서 <모듈>.ts 추출 후 import 연결 + skip 해제` 한 줄 명시.
- **검증**: `npm run test:unit` 전부 통과(todo/skip은 통과로 집계, 실제 단정은 4-a 전부 그린) + `npx tsc --noEmit` 0(테스트 파일 타입도 통과). **커밋**: `test(fe): pure-function unit tests + forward stubs (CP230 step4)`.

### Step 5 — 로컬 dev 서버 띄워 전체 실행(통합 확인)
1. `start_demo.ps1` 기동 상태 확인(백엔드 8000 ready, 프론트 3000 ready).
2. 전체 스위트 실행:
   ```powershell
   npm run test:unit
   npm run test:e2e
   ```
3. 결과 표를 `docs/cp230_report.md` 에 기록(통과 수, baseline 파일 목록, 2회 재현 결과).
- **검증**: 두 명령 모두 그린. **커밋**(보고서/문서만): `docs(fe): CP230 characterization report (CP230 step5)`.

---

## 인터페이스 보존

- **프로덕션 코드 signature·동작 불변.** 이 CP는 `frontend/src/**` 의 export 함수 시그니처·반환·API 응답 schema·React props를 **하나도 바꾸지 않는다**. 추가되는 건 테스트 파일(`*.test.ts`, `tests/e2e/**`)과 config(`playwright.config.ts`, `vitest.config.ts`)와 `package.json` scripts 항목뿐.
- `package.json` 의 기존 `scripts.dev/build/start` 와 기존 deps/devDeps는 **삭제·수정 금지, 추가만**.
- 만약 테스트 작성 중 "현재 동작이 버그처럼 보인다"는 판단이 들어도 **고치지 말고 현재 동작 그대로 고정**하고, 의심점은 보고서 "후속"에 적는다(동작 변경은 별도 기능 CP).
- 결정론화가 프로덕션 코드 수정을 요구하면(예: `dateUtils.buildDefaultPriceWindow` 를 주입식으로 바꿔야 스크린샷이 안정) → **호출자 영향 분석 후 차단 보고**. 이 CP에서 임의로 바꾸지 않는다(스크린샷은 마스킹으로 우회하는 게 1순위).

---

## 성공 기준 (측정 가능)

| 항목 | 시작 | 목표 |
|---|---|---|
| Playwright smoke (4화면) | 0 | 4/4 통과, `pageerror`/`console.error` 0 |
| Screenshot baseline | 0 | 4화면 baseline png 생성, **2회 연속 재현(diff 0)** |
| Vitest 순수 함수 테스트(4-a) | 0 | 5개 대상 파일, 분기 경계 포함 전부 통과 |
| 전방 stub(4-c) | 0 | 3파일 `todo/skip` 명문화(추출 시 연결 가능) |
| `npx tsc --noEmit` 추가 에러 | 0 | 0 (테스트 파일 포함) |
| 회귀(프로덕션 동작 변경) | — | 0 (코드 미이동) |
| 예상 시간 | — | 3~4시간 |

---

## 검증 (구체 명령 / 기대 결과)

> 모두 `frontend` 디렉토리에서. 선행으로 루트에서 `start_demo.ps1` 기동.

1. 백엔드+프론트 기동:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
   ```
   기대: `백엔드 ready ...`, `프론트 ready ...` 출력 후 `exit 0`. (안 뜨면 → 차단 트리거.)
2. 타입:
   ```powershell
   npx tsc --noEmit
   ```
   기대: 출력 없음(에러 0).
3. 단위:
   ```powershell
   npm run test:unit
   ```
   기대: 4-a 케이스 전부 `passed`, 4-c는 `todo/skipped`, 실패 0.
4. e2e + 스크린샷(2회 연속, 재현성 확인):
   ```powershell
   npm run test:e2e
   npm run test:e2e
   ```
   기대: 두 번 다 4 smoke + 스크린샷 `passed`. 1·2회 사이 diff 0.
5. baseline 산출물 확인(Glob/디렉토리 나열로):
   - `frontend/tests/e2e/**/__screenshots__/report*.png` 등 4개 존재.

---

## 차단 트리거 (중요)

> 다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **화면 로드 자체 실패(데모 깨짐)**: 4화면 중 하나라도 `pageerror`/`console.error` 발생, 흰 화면, 헤더 셀렉터 미가시, 또는 `start_demo.ps1` 가 ready에 도달 못함 → **멈추고 보고**(어느 화면, 콘솔/네트워크 로그, 백엔드 stderr `logs/backend_dev.err.log` tail 첨부). 이건 안전망 이전에 데모가 깨진 신호다.
2. **스크린샷 비결정(매번 다름)**: 마스킹·`animations:"disabled"`·`networkidle` 까지 했는데도 `test:e2e` 2회 사이 diff 발생 → **멈추고 보고**(diff난 화면 + 어떤 영역인지). 추가 마스킹으로 못 잡으면 baseline 강행 금지.
3. **순수 함수 단정이 현재 동작과 불일치**: 내가 "기대값"이라 적은 게 실제 코드 출력과 다르면 → 코드를 고치지 말고 **현재 출력으로 단정을 맞춘 뒤**, "현재 동작이 이상함" 의심을 보고. (특히 `formatRatio` 경계 `|value|<=1`, `evaluateFollowReturn` 방어 우위 분기 등.)
4. **로케일/TZ 의존으로 단위 테스트가 머신마다 흔들림**: `Intl` 결과가 config로 고정 안 되면 → 표기 차이는 완화 단정으로 두되, 값 의미까지 흔들리면 **보고**.
5. **결정론화에 프로덕션 코드 수정이 필요**: 스크린샷/단위 안정화를 위해 `frontend/src/**` 동작을 바꿔야만 하는 상황 → **차단 보고**(호출자 영향 분석 첨부). 이 CP에서 동작 변경 금지.
6. **기존 빌드/타입 회귀**: config 추가만 했는데 `npx tsc --noEmit` 또는 `npm run build` 가 새로 깨짐 → 멈추고 보고.
7. **사용자 직접수정 흔적**: 대상 파일에 진행 중인 사용자 수정/충돌 흔적 → revert 금지, 멈추고 보고.
8. **포트 충돌/이중 기동**: Playwright가 3000을 또 띄우려 하거나 8000/3000 점유 충돌 → 멈추고 보고(config의 `webServer` 미사용 원칙 위반 점검).

---

## ADR

완료 후 `docs/adr/0020-frontend-characterization-playwright-vitest.md` 1장(200~300단어) 작성.
기록할 것: **왜 프론트 분리(CP231~234) 전에 Playwright(로드 smoke + 스크린샷 baseline) + Vitest(순수 함수)를 안전망으로 택했는지**, 스크린샷 결정론을 위해 `ReportView`를 기준 화면으로 삼고 나머지는 마스킹/`networkidle`/애니메이션 비활성으로 우회한 결정, `webServer` 미사용(기존 `start_demo.ps1` 서버 재사용)·chromium 1종·`node` 환경(jsdom 미설치) 선택, 그리고 전방 stub(`it.todo`)로 CP231~233 추출 회귀를 예약한 결정.

---

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다.

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ___
- **[구조 결함]** PASS / WARN / FAIL — 사유: ___
- **[모델 영향]** PASS / WARN / FAIL — 사유: ___ (이 CP는 학습/calibration 미접촉이라 통상 PASS)

---

## 산출물

- **변경/추가 파일**:
  - `frontend/playwright.config.ts` (신규)
  - `frontend/vitest.config.ts` (신규)
  - `frontend/package.json` (scripts 4개 추가: `test:e2e`, `test:e2e:update`, `test:unit`, `test:unit:watch`; devDeps 추가)
  - `frontend/.gitignore` (test-results/report ignore, baseline 제외)
  - `frontend/tests/e2e/screens.smoke.spec.ts` (신규)
  - `frontend/tests/e2e/**/__screenshots__/*.png` (baseline 4종, 커밋)
  - `frontend/src/lib/backtest/evaluators.test.ts`
  - `frontend/src/lib/backtest/predictionHelpers.test.ts`
  - `frontend/src/lib/formatters.test.ts`
  - `frontend/src/lib/training/formatters.test.ts`
  - `frontend/src/lib/staleness.test.ts`
  - `frontend/src/lib/backtest/simulationEngine.test.ts` (stub, CP231 예약)
  - `frontend/src/lib/backtest/signalBuilder.test.ts` (stub, CP232 예약)
  - `frontend/src/lib/training/experimentBuilder.test.ts` (stub, CP233 예약)
  - `docs/adr/0020-frontend-characterization-playwright-vitest.md`
- **보고서**: `docs/cp230_report.md` — 요구 / 한 일 / 결정 / 후속(필요한 만큼만). 후속에 4-c stub 연결 위치(CP231~233)와 결정론화 미해결 의심점을 적는다.
