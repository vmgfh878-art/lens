# CP224b 보고서 — Dead Code 검출 + 안전 제거

- **상태**: ✅ 완료
- **기간**: 2026-06-03
- **모드**: code
- **선행 의존**: CP222 ✅ / CP223 ✅
- **마지막 그린 커밋**: e526a8c (Step 1 F401)

---

## 요구

dead code를 자동 도구로 후보 산출 → **각 후보를 grep으로 호출 0 직접 확인한 것만** 제거. 광범위 일괄 삭제 금지. 도구 출력 = 제안. 운영 코드 동작 보존.

---

## Step 1 — ruff F401 (커밋 `e526a8c`)

- backend/app 전체에서 F401 검출 **1건만**: `backend/app/routers/v1/stocks.py:11` `PredictionData` unused import.
- re-export 검증: `from app.routers.v1.stocks import PredictionData` 검색 0건 + `__all__` 없음 → 진짜 unused.
- fix 후: `ruff check backend/app --select F401` → All checks passed.
- 회귀: pytest backend/tests 87 passed (CP223 baseline 동일).

---

## Step 2 — vulture --min-confidence 60 후보 (75건)

명령: `python -m vulture backend/app --min-confidence 60 --sort-by-size`

**총 75건 검출.** 차단 트리거 9에 따라 일괄 제거 금지. 분류 + 상위 진짜 후보만 Step 4에서 grep 검증.

### 분류 A — FastAPI router/handler (오탐, 데코레이터 등록)

라우터 함수 18개. 모두 `@router.get/post/...` 또는 `@app.exception_handler/on_event` 데코레이터 등록 → 호출자가 코드상 안 보여도 사용 중. **무시.**

| 파일:줄 | 심볼 | 데코레이터 |
|---|---|---|
| routers/v1/stocks.py:36 | list_stocks | `@router.get("")` |
| routers/v1/stocks.py:52 | get_prices | `@router.get("/{ticker}/prices")` |
| routers/v1/stocks.py:79 | get_indicators | `@router.get("/{ticker}/indicators")` |
| routers/v1/stocks.py:96 | get_product_prediction_history | `@router.get("/{ticker}/predictions/product-history")` |
| routers/v1/predictions.py:107 | list_tickers | `@router.get(...)` |
| routers/v1/predictions.py:123 | get_line | `@router.get("/line/{ticker}")` |
| routers/v1/predictions.py:153 | get_band_1d | `@router.get("/band/1d/{ticker}")` |
| routers/v1/predictions.py:187 | get_band_1w | `@router.get("/band/1w/{ticker}")` |
| routers/v1/strategies.py:15 | list_strategies | `@router.get("")` |
| routers/v1/strategies.py:41 | scan_strategy | `@router.get(...)` |
| routers/v1/strategies.py:54 | backtest_strategy_ticker | `@router.get(...)` |
| routers/v1/health.py:13 | live | `@router.get("/live")` |
| routers/v1/health.py:28 | ready | `@router.get("/ready")` |
| routers/v1/ai.py:195 | list_ai_runs | `@router.get(...)` |
| routers/v1/ai.py:223 | get_ai_run | `@router.get(...)` |
| routers/v1/ai.py:239 | list_run_evaluations | `@router.get(...)` |
| routers/v1/ai.py:255 | list_run_backtests | `@router.get(...)` |
| routers/v1/admin.py:47 | reload_v1_predictions | `@router.post(...)` |
| routers/v1/admin.py:74 | debug_state | `@router.get(...)` |
| main.py:86 | health_check | `@app.get("/")` |
| main.py:66 | _load_v1_predictions_cache | `@app.on_event("startup")` |
| main.py:97 | handle_app_error | `@app.exception_handler(AppError)` |
| main.py:111 | handle_validation_error | `@app.exception_handler(RequestValidationError)` |
| main.py:124 | handle_value_error | `@app.exception_handler(ValueError)` |
| main.py:136 | handle_unexpected_error | `@app.exception_handler(Exception)` |

**판정: 25 오탐 (등록 데코레이터로 사용 중)**

### 분류 B — Supabase 관련 (보류, 무시)

| 파일:줄 | 심볼 |
|---|---|
| db.py:38 | reset_supabase_client |
| repositories/market_repo.py:97 | fetch_price_rows |
| repositories/market_repo.py:146 | fetch_indicator_rows |
| repositories/market_repo.py:335 | fetch_stocks |

**판정: 4 보류 (Supabase, 사용자 재연결 예정)**

### 분류 C — Pydantic schema 필드 (오탐, response_model 직렬화)

`schemas/stocks.py`의 unused variable 36개는 모두 Pydantic model field. JSON 직렬화 시 사용. vulture가 명시 호출 없으면 unused로 잘못 판정.

| 파일:줄 | 심볼 | 비고 |
|---|---|---|
| schemas/stocks.py:10 | sector | StockSummary 필드 |
| schemas/stocks.py:11 | industry | StockSummary 필드 |
| schemas/stocks.py:12 | market_cap | StockSummary 필드 |
| schemas/stocks.py:21 | volume | PriceBar 필드 |
| schemas/stocks.py:35-43 | macd_ratio, bb_position, ma_5_ratio, ma_20_ratio, ma_60_ratio, vol_change, volume, atr_ratio, regime_label | IndicatorPoint 필드 |
| schemas/stocks.py:57-68 | asof_date, decision_time, model_ver, signal, forecast_dates, upper_band_series, lower_band_series, conservative_series, line_series, band_quantile_low, band_quantile_high | PredictionData 필드 |
| schemas/stocks.py:52 | PredictionData (class) | 클래스 자체도 unused로 떴지만 schemas 모듈에서 export, F401 처리됨 |
| schemas/stocks.py:73-104 | asof_date, display_horizon, line_run_id, band_run_id, date_range, row_count, line_history, band_history, manifest_summary, empty_reason | 다른 response_model 필드 |

**판정: 36 오탐 (Pydantic field, response_model 직렬화 사용)**. 단 `PredictionData` 클래스(stocks.py:52)는 다른 곳에서 import 없음 검증 필요(Step 4-grep).

### 분류 D — 진짜 검증 필요 (Step 4 grep 대상)

| 파일:줄 | 심볼 | 종류 | confidence | 판정 |
|---|---|---|---|---|
| services/strategy_backtest_svc.py:71 | _safe_gt | function | 60% | grep TBD |
| services/strategy_backtest_svc.py:75 | _safe_gte | function | 60% | grep TBD |
| services/strategy_backtest_svc.py:79 | _safe_lt | function | 60% | grep TBD |
| services/strategy_backtest_svc.py:83 | _safe_lte | function | 60% | grep TBD |
| services/feature_svc.py:125 | default_horizon_for_timeframe | function | 60% | grep TBD |
| services/feature_svc.py:375 | build_price_features | function | 60% | grep TBD |
| services/feature_svc.py:575 | build_latest_feature_rows | function | 60% | grep TBD |
| services/model_svc.py:22 | normalize_model_name | function | 60% | grep TBD |
| services/model_svc.py:52 | resolve_horizon | function | 60% | grep TBD |
| services/parquet_store.py:49 | require | function | 60% | grep TBD |
| services/product_prediction_history_svc.py:83 | mtime | variable | 100% | grep TBD |
| strategies/strategy_rules.py:96 | list_strategies | function | 60% | grep TBD (라우터의 list_strategies와 별개 모듈) |
| core/exceptions.py:62 | InvalidRunStatusError | class | 60% | grep TBD |
| core/exceptions.py:72 | InsufficientHistoryError | class | 60% | grep TBD |
| schemas/stocks.py:52 | PredictionData (class) | 60% | grep TBD |

**판정 합계: 분류 A·B·C 65개 자동 제외 + 분류 D 15개 grep 검증 대상.**

---

---

## Step 3 — ts-prune 후보 (76건)

명령: `cd frontend; npx ts-prune` (일회성, 영구 devDep 추가 없음).

### 분류 E — `(used in module)` 50건: export만 불필요, 코드는 사용 중

`export` 키워드만 제거 가능 (안전). 그러나 일괄 제거는 위험 → 별도 청소 CP로 분리 권장. 본 CP는 dump만.

대표:
- `src/components/{IndicatorPanel,StatusInline}.tsx` props 타입
- `src/lib/apiErrors.ts` 내부 타입 (ApiErrorShape, ApiErrorKind, extractHttpStatus)
- `src/lib/dateUtils.ts:37 formatDate`
- `src/lib/productSlots.ts` 3 타입 (Kind/Status/RefreshPolicy)
- `src/lib/staleness.ts:100 BandStaleness`
- `src/lib/v1Adapter.ts` 4 함수 (normalizeBandValueToPrice 등)
- `src/lib/chart/utils.ts` 9 함수
- `src/lib/training/` 다수 (detailFields, lineTimeline, reproducibility, runUtils, staticEvaluation, usageData)

→ **50건 모두 "export-only 제거" 후보, 별도 청소 CP 권장**. 본 CP에서는 grep 검증 없이 batch 제거 금지.

### 분류 F — 완전 unused, 도구 오탐 가능 (검토 필요)

| 파일:줄 | 심볼 | 비고 |
|---|---|---|
| playwright.config.ts:16 | default | Playwright가 자동 import → 오탐 |
| vitest.config.ts:12 | default | Vitest가 자동 import → 오탐 |
| .next/types/app/layout.ts:49 | PageProps | Next.js build 결과 → 오탐 |
| .next/types/app/layout.ts:53 | LayoutProps (used in module) | 동상 |
| .next/types/app/page.ts:49 | PageProps (used in module) | 동상 |
| .next/types/app/page.ts:53 | LayoutProps | 동상 |

→ **6 오탐 (config/build 자동 생성)**.

### 분류 G — 진짜 검증 필요 (Step 4 grep 대상)

| 파일:줄 | 심볼 | 종류 | 판정 |
|---|---|---|---|
| src/api/client.ts:6 | api | namespace export | grep TBD |
| src/api/client.ts:6 | getBackendBaseUrl | function | grep TBD |
| src/api/client.ts:9 | ApiMeta | type | grep TBD |
| src/api/client.ts:10 | ApiResponse | type | grep TBD |
| src/api/client.ts:16 | PriceResult | type | grep TBD |
| src/api/client.ts:17 | IndicatorResult | type | grep TBD |
| src/api/client.ts:22 | ProductPredictionHistoryManifestSummary | type | grep TBD |
| src/api/client.ts:23 | ProductPredictionHistoryResult | type | grep TBD |
| src/api/client.ts:24 | V1BandPredictionPoint | type | grep TBD |
| src/api/client.ts:26 | V1LinePredictionPoint | type | grep TBD |
| src/api/client.ts:31 | AiRunStatus | type | grep TBD |
| src/api/client.ts:33 | BacktestSummary | type | grep TBD |
| src/api/client.ts:34 | EvaluationSummary | type | grep TBD |
| src/api/client.ts:37 | StrategyBacktestResult | type | grep TBD |
| src/api/client.ts:38 | StrategyPortfolioMetrics | type | grep TBD |
| src/api/client.ts:39 | StrategyScanResult | type | grep TBD |
| src/api/client.ts:48 | fetchProductPredictionHistory | function | grep TBD |
| src/api/client.ts:56 | fetchRunBacktests | function | grep TBD |
| src/api/client.ts:57 | fetchRunEvaluations | function | grep TBD |
| src/components/training/ReproducibilitySection.tsx:13 | default | component | grep TBD |
| src/components/training/UsageDataSection.tsx:12 | default | component | grep TBD |
| src/lib/training/lineTimeline.ts:48 | LineExperimentCategory | type | grep TBD |
| src/lib/training/lineTimeline.ts:49 | LineExperimentNode | type | grep TBD |

→ **23 진짜 검증 후보, Step 4에서 grep**.

**판정 합계 (ts-prune)**:
- 분류 E 50: 별도 청소 CP 권장 (export-only 정리, 안전하지만 큰 변경)
- 분류 F 6: 도구 오탐 (config/Next.js)
- 분류 G 23: Step 4 grep 대상

---

---

## Step 4 — 후보 grep 검증

### 백엔드 D분류 15 → 확정 dead 12 + 오탐 2 + lru_cache 1

| 심볼 | grep 결과 | 판정 |
|---|---|---|
| _safe_gt/_gte/_lt/_lte (strategy_backtest_svc) | 자기 파일 정의 4건만, 호출 0 | **확정 dead 4** |
| default_horizon_for_timeframe (feature_svc:125) | 외부 호출 0 | **확정 dead** |
| build_price_features (feature_svc:375) | `ai/preprocessing.py:929` 호출 | **오탐, 사용 중** |
| build_latest_feature_rows (feature_svc:575) | 외부 호출 0 | **확정 dead** |
| normalize_model_name (model_svc:22) | 외부 호출 0 | **확정 dead** |
| resolve_horizon (model_svc:52) | 외부 호출 0 | **확정 dead** |
| require (parquet_store:49) | 외부 호출 0 | **확정 dead** |
| mtime (product_prediction_history_svc:83) | `@lru_cache` 인자 → cache key | **오탐, 사용 중** |
| list_strategies (strategy_rules:96) | 라우터의 list_strategies는 별개 함수 | **확정 dead** |
| InvalidRunStatusError (exceptions:62) | `raise InvalidRunStatusError` 0건 | **확정 dead** |
| InsufficientHistoryError (exceptions:72) | `raise InsufficientHistoryError` 0건 | **확정 dead** |
| PredictionData (schemas/stocks:52) | 외부 import 0 (Step 1 F401 후) | **확정 dead** |

### 프론트 G분류 23 → 확정 dead 4 + 오탐 19

| 그룹 | 심볼 | grep 결과 | 판정 |
|---|---|---|---|
| client.ts | api/getBackendBaseUrl/ApiMeta/ApiResponse/PriceResult/IndicatorResult 등 19개 | **18 파일이 `@/api/client`에서 import** (BacktestView, StockView, TrainingView, lib/*.ts 등) | **모두 오탐, ts-prune이 re-export aggregator 잘못 분류** |
| 컴포넌트 | ReproducibilitySection.tsx default | `training/ReproducibilitySection` import 0 | **확정 dead 파일** |
| 컴포넌트 | UsageDataSection.tsx default | `training/UsageDataSection` import 0 | **확정 dead 파일** |
| 타입 alias | LineExperimentCategory, LineExperimentNode (lineTimeline:48,49) | 자기 파일만 (CP218 호환 alias 주석, 다른 import 0) | **확정 dead 2 alias** |

---

## Step 5 — 확정 dead 제거 + 회귀 검증

### 백엔드 (12 dead 제거)
- `backend/app/services/strategy_backtest_svc.py`: `_safe_gt/_gte/_lt/_lte` 4 함수 제거.
- `backend/app/services/feature_svc.py`: `default_horizon_for_timeframe` + `build_latest_feature_rows` 2 함수 제거.
- `backend/app/services/model_svc.py`: `normalize_model_name` + `resolve_horizon` 2 함수 제거.
- `backend/app/services/parquet_store.py`: `require` 함수 제거.
- `backend/app/schemas/stocks.py`: `PredictionData` 클래스 제거 (+ chain F401 fix로 `Any, Field` import 제거).
- `backend/app/strategies/strategy_rules.py`: `list_strategies` 함수 제거.
- `backend/app/core/exceptions.py`: `InvalidRunStatusError` + `InsufficientHistoryError` 클래스 제거.

### 프론트 (4 dead 제거)
- `frontend/src/components/training/ReproducibilitySection.tsx` 파일 삭제.
- `frontend/src/components/training/UsageDataSection.tsx` 파일 삭제.
- `frontend/src/lib/training/lineTimeline.ts`: `LineExperimentCategory` + `LineExperimentNode` alias 제거 (CP218 호환).

### 회귀 검증 (모두 ✅)
- pytest backend/tests: **87 passed** (CP223 baseline 78+9 동일, 신규 실패 0).
- ruff check backend/app --select F401: **0** (chain F401 2건 자동 fix 포함).
- CP223 9 snapshot: 그대로 통과 (응답 schema 무변경).
- frontend tsc --noEmit: **0 에러**.
- Vitest: **107 passed / 12 todo / 0 failed**.
- Playwright e2e: **4 passed (9.2s, screenshot diff 0)**.

---

## Step 5 후속/잔여

### 별도 청소 CP 권장
- **ts-prune 분류 E (50건)**: `(used in module)` 항목. `export` 키워드만 제거해도 안전. 별도 일괄 청소 CP에서.
  - 대표: `lib/apiErrors.ts`(ApiErrorShape/ApiErrorKind/extractHttpStatus), `lib/dateUtils.ts:37 formatDate`, `lib/productSlots.ts` 3 타입, `lib/v1Adapter.ts` 4 함수, `lib/chart/utils.ts` 9 함수, `lib/training/*` 다수, `components/{IndicatorPanel,StatusInline}.tsx` props 타입.

### 보류 (Supabase 정책)
- `db.py:38 reset_supabase_client` (Supabase 재연결 예정)
- `repositories/market_repo.py:97,146,335` fetch_price_rows / fetch_indicator_rows / fetch_stocks (Supabase 미연결 시 호출 0이지만 재연결 시 부활)

### 오탐 (기록만)
- vulture 65건: FastAPI router/handler 25 + Pydantic field 36 + Supabase 4.
- ts-prune 25건: config/Next.js 6 + client.ts re-export 19.

---

## 성공 기준 충족표

| 항목 | 기준 | 실측 | 결과 |
|---|---|---|---|
| ruff F401 violations | 0 | 0 (Step 1 + chain) | ✅ |
| vulture 후보 문서화 | 전체 표 | 75건 분류 A~D | ✅ |
| ts-prune 후보 문서화 | 전체 표 | 76건 분류 E~G | ✅ |
| 확정 dead 제거 | grep 확인분만 | 백엔드 12 + 프론트 4 | ✅ |
| 백엔드 pytest 회귀 | 0 | 87 passed (78+9 baseline) | ✅ |
| CP223 snapshot diff | 0 | 0 | ✅ |
| 프론트 tsc --noEmit | 0 | 0 | ✅ |
| 프론트 e2e | 4 passed | 4 passed (diff 0) | ✅ |
| mypy 신규 error | 0 추가 | 0 | ✅ |

---

## 자가 점검

- **[Plan v3 정합]** **PASS** — read-path만 손댐. 모델·calibration·EODHD·밴드 파이프라인 무관. Plan v3 결정 (α=1/β=2, fidelity, EODHD 유지) 영향 0.
- **[구조 결함]** **PASS** — dead code 제거로 결합도 감소. CP223 snapshot diff 0이 schema 보존 증명. Supabase·테스트·collector 명시 제외로 안전 경계 유지. 오탐 65 + 25 + 보류 4는 보고서에 명시.
- **[모델 영향]** **PASS** — 학습/추론/calibration/RevIN/dropout 코드 미변경. snapshot diff 0 → 모델 출력 불변. parquet read-only.
