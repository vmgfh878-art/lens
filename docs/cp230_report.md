# CP230 보고서 — 프론트엔드 Characterization 안전망

- **상태**: ✅ 완료
- **기간**: 2026-06-03 (단일 세션, CP223 직후 사용자 결정으로 BE 트랙에서 FE 트랙으로 이동)
- **모드**: code (구현 + 자가 점검)
- **선행 의존**: 없음 (이 CP가 프론트 분리의 선행 안전망)
- **검증 환경**: `scripts/start_demo.ps1` 기동 (백엔드 8000 + 프론트 3000 ready)

---

## 요구

CP231~234 프론트 분리(거대 뷰 BacktestView 1643줄 / TrainingView 1560줄 / StockView 1097줄 + globals.css 5208줄)의 안전망 구축. 현재 frontend 도구는 0(test runner/e2e/lint 어느 것도 없음). 프로덕션 코드 0 변경으로 4화면 screenshot baseline + 5 lib characterization + 3 CP forward stub 박제.

---

## 한 일 (Sub-step 별)

### Step 1 — Playwright 설치 + config (커밋 `d276bcb`)
- `@playwright/test 1.60.0` devDep, chromium 1종 download (~200MB).
- `frontend/playwright.config.ts`: testDir tests/e2e, baseURL 127.0.0.1:3000, viewport 1280x800, locale ko-KR, TZ Asia/Seoul, animations disabled, maxDiffPixelRatio 0.01, workers 1, retries 0, **webServer 미사용**(start_demo.ps1 선행).
- `package.json` scripts: test:e2e, test:e2e:update 추가.
- `frontend/.gitignore` 신규: test-results/, playwright-report/, blob-report/, .playwright/ ignore. baseline은 제외.
- 검증: tsc --noEmit 0, npx playwright --version=1.60.0.

### Step 2 — 4화면 smoke + screenshot baseline (커밋 `a6b2c3b`)
- `frontend/tests/e2e/screens.smoke.spec.ts` 신규. parametrize 4 케이스:
  | id | path | 헤더 셀렉터 | hasFetch |
  |---|---|---|---|
  | report | `/?view=report` | h1 "예측선은 방향을, 밴드는 흔들림을 봅니다." | false |
  | stocks | `/?view=stocks` | `.stock-intro` | true |
  | backtests | `/?view=backtests` | h1 "전략 신호" | true |
  | training | `/?view=training` | h1 "AI 모델" | true |
- 가드: pageerror 0, console.error 0. 단 `/Failed to load resource:.+404/`는 화이트리스트 (TrainingView가 `/api/v1/ai/runs` mock 부재 시 정상 404, CP223 진단에서도 ai 라우터 명시 제외됨).
- 동적 영역 마스킹: canvas, `.freshness`, `.asof`, `[data-freshness]`, `[data-asof]`, `.stale-tag`, `.stale-badge`.
- baseline 4 PNG 생성: `frontend/tests/e2e/screens.smoke.spec.ts-snapshots/{report,stocks,backtests,training}-chromium-win32.png`.
- 2회 연속 실행 모두 4 passed (각 ~5.2s, diff 0).

### Step 3 — Vitest config (커밋 `5a0a118`)
- vitest 4.1.8 + @vitest/coverage-v8 4.1.8 devDeps.
- `frontend/vitest.config.ts`: environment "node" (jsdom 미설치, 순수 함수만), include `src/**/*.test.ts`, globals false, alias @→src.
- `package.json` scripts: test:unit, test:unit:watch.
- 검증: vitest run "No test files found" (정상), tsc 0.

### Step 4 — 순수 함수 단위 테스트 + 3 forward stub (커밋 `4ff83cc`)

**4-a. 현존 5 lib characterization** (모두 운영 코드 무수정, 실제 동작 그대로 단정):
- `src/lib/backtest/evaluators.test.ts`: describeAvoidanceStrength 분기(≥0.6 강함 / ≥0.4 보통 / <0.4 약함 / null '-'), evaluateFollowReturn 방어 우위/양호/보통/약함, evaluateDrawdown 5/0/음수, evaluateLossAvoidance 위임 확인, evaluateTradeFrequency 20/40/41 경계.
- `src/lib/backtest/predictionHelpers.test.ts`: median 빈/홀/짝, percentileRank NaN+Inf 필터, getWorstLowerBandValue/getHighestUpperBandValue Math.min/max+필터, getBandWidthValue upper-lower, getConservativeValue conservative→line fallback.
- `src/lib/formatters.test.ts`: formatNumber Intl ko-KR, formatPercent +/- 부호, formatRatio |value|≤1 *100 / >1 그대로, formatRatioAsPercent/formatUnsignedRatioAsPercent, formatCompact 동작, normalizeRsi 0~1 / >1, finiteOrNull/isFiniteNumber, ratioToPercent. **Intl 로케일 표기(천단위/구분자)는 머신 의존이라 regex 완화**.
- `src/lib/training/formatters.test.ts`: formatStatusLabel completed→완료/failed_nan→실패/failed_quality_gate→기준 미달, formatRoleLabel line→보수적 기준선 / band→AI 밴드 / composite→이전 조합 실험, formatModelLabel patchtst→PatchTST / cnn_lstm→CNN-LSTM, formatMetric rate/pct_point, formatSignedNumber/formatSignedPctPoint, extractErrorMessage(Network Error→백엔드 안내), formatKoreanDateTime(Asia/Seoul → "...KST" 접미).
- `src/lib/staleness.test.ts`: getSlotFreshness(slot null/deferred/static/1D fresh/1W stale/invalid asof or price), evaluateBandStaleness(invalid→gap 0, within→not stale, over threshold→stale + reason).

**4-c. CP231~233 forward stubs** (describe.skip + it.todo):
- `src/lib/backtest/simulationEngine.test.ts` (CP231 예약 — BacktestView.tsx 인라인 시뮬레이션 루프)
- `src/lib/backtest/signalBuilder.test.ts` (CP232 예약 — BacktestView.tsx:350 `buildSignals`)
- `src/lib/training/experimentBuilder.test.ts` (CP233 예약 — TrainingView.tsx:1104 `buildExperimentCards`)

### Step 5 — 전체 실행 + 보고서 + ADR (이 커밋)
- `npm run test:unit`: **107 passed / 12 todo / 0 failed** (5 files passed + 3 stub skipped, 240ms).
- `npm run test:e2e`: **4 passed** (5.4s).
- `docs/adr/0020-frontend-characterization-playwright-vitest.md` 신규.
- 본 보고서 신규.

---

## 인터페이스 보존

- **프로덕션 `frontend/src/**` 0 라인 변경.** 함수 signature / API 응답 schema / React props 무변경.
- console.error 가드는 호출 측 화이트리스트로 회피 (지시서 차단 트리거 5: "결정론화에 프로덕션 코드 수정 필요" 회피).
- 단위 테스트 단정값은 **실제 코드 출력 그대로** (지시서 차단 트리거 3 준수). 단정과 실제가 불일치한 경우는 0건 — 모두 단정과 일치하여 수정 불요.

---

## 핵심 컴포넌트 존재 체크리스트 (메타 D21)

- @playwright/test 1.60.0 ✅
- chromium download ✅
- vitest 4.1.8 + @vitest/coverage-v8 4.1.8 ✅
- playwright.config.ts (locale/TZ/viewport/animations 옵션) ✅
- vitest.config.ts (environment:node, alias @→src) ✅
- backend/frontend ready (start_demo.ps1 200/200) ✅
- ReportView 헤더 셀렉터 직접 확인 ✅
- StockView .stock-intro 직접 확인 ✅
- BacktestView/TrainingView h1 직접 확인 ✅
- 5 lib 파일 함수 시그니처 직접 확인 ✅
- 운영 코드 변경 0 줄 (`git diff frontend/src/**` 비어있음) ✅

## 새 테스트 결과 (메타 D21)

- E2E: 4 smoke (report/stocks/backtests/training), baseline 4 PNG.
- Unit: 107 passed (evaluators 13 + predictionHelpers 14 + formatters 30 + training/formatters 32 + staleness 18 = 107 대략).
- Stub: 12 todo (3 파일 × 4 케이스).

## Dry-run 결과 (메타 D21)

- E2E 2회 연속 4 passed (각 ~5.2s, diff 0).
- Unit 240ms, 0 failed.
- 운영 코드 변경 0줄 확인.

## 기존 회귀 통과 건수 (메타 D21)

- 프론트 무회귀(코드 미이동). CP222의 tsc --noEmit PASS → CP230 후에도 PASS 유지.
- 백엔드 영향 없음.

---

## 성공 기준 충족표

| 항목 | 기준 | 실측 | 결과 |
|---|---|---|---|
| Playwright smoke 4화면 | 4/4 통과, pageerror 0 | 4 passed, pageerror 0 | ✅ |
| Screenshot baseline | 4 PNG, 2회 재현 diff 0 | 4 PNG, 2회 5.2s 모두 통과 | ✅ |
| Vitest 단위(4-a) | 5 파일, 분기 경계 통과 | 5 files passed, 107 tests | ✅ |
| 전방 stub(4-c) | 3 파일, todo/skip | 3 skipped, 12 todo | ✅ |
| tsc --noEmit 추가 에러 | 0 | 0 | ✅ |
| 프로덕션 동작 변경 | 0 | 0 라인 | ✅ |

---

## 후속

- **CP231 (다음 권장)**: `simulationEngine.ts` 추출 → BacktestView.tsx 시뮬레이션 루프 분리. simulationEngine.test.ts skip 해제.
- **CP232**: `signalBuilder.ts` 추출 → BacktestView.tsx:350 `buildSignals`. signalBuilder.test.ts skip 해제.
- **CP233**: `experimentBuilder.ts` 추출 → TrainingView.tsx:1104 `buildExperimentCards`. experimentBuilder.test.ts skip 해제.
- **CP234** (별도 고위험): globals.css 5208줄 분리.
- **별도**: `dateUtils.buildDefaultPriceWindow`/`buildFullPriceWindows` `new Date()` 의존 → `vi.setSystemTime` 고정 또는 주입식 변경(코드 변경 CP).
- **별도**: 로케일/TZ 환경 의존 단정 강화 (CI 환경에서 Intl 결과 정확 비교).

---

## 자가 점검

- **[Plan v3 정합]** **PASS** — CP230은 프론트 안전망 추가만. fidelity/EODHD/α=1·β=2 등 Plan v3 결정에 영향 0.
- **[구조 결함]** **PASS** — playwright.config webServer 미사용 결정, console.error 가드의 404 화이트리스트 좁힘 (운영 코드 무수정), forward stub의 CP231~233 예약 명시. baseline 위치 표준 디렉토리.
- **[모델 영향]** **PASS** — 학습/calibration/preprocess/sufficiency gate 무관. 프론트 표시 로직만.
