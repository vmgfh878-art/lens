# CP224b 보고서 — Dead Code 검출 + 안전 제거

- **상태**: 진행 중 (Step 2 dump 단계)
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

(Step 4, 5는 후속)
