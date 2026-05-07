# CP147-G 제품 모델 저장/선택 계약 감사

작성일: 2026-05-07

## 1. Executive Summary

최종 판정: **WARN**

현재 제품 화면에 연결된 1D line, 1D band, 1W line은 의도한 역할과 대체로 맞다. 1W band는 아직 제품 후보 없음/준비 중으로 고정되어 있고, 1M은 AI 예측 비활성/가격 전용으로 유지된다. `ai.inference --save`의 bulk 저장 기본 차단과 `save_product_latest_predictions()`의 latest-only guard도 존재한다.

다만 제품 모델 registry가 아직 없다. 제품 run id는 `frontend/src/components/StockView.tsx`, `frontend/src/components/TrainingView.tsx`, `frontend/src/components/BacktestView.tsx`, 여러 리허설 스크립트, `product_prediction_history_1D.manifest.json`에 수동으로 박혀 있다. 또한 band calibration은 리포트/분석용 경로에는 있으나 현재 제품 inference 저장 경로에서 calibration scale/shift를 자동 적용하는 증거가 없다. 따라서 calibrated 지표를 product stored band 성능으로 말하면 안 된다.

핵심 결론:

- LM 제품 후보는 `checkpoint_selection=line_gate`와 `role=line_model`이 함께 맞아야 한다.
- BM 제품 후보는 `checkpoint_selection=band_gate`와 `role=band_model`이 함께 맞아야 한다.
- `checkpoint_selection=val_total` 기본값 run은 `completed`라도 제품 후보로 자동 승격하지 않는다.
- `failed_quality_gate`, `failed_nan`은 제품 후보에서 제외한다.
- 현재 제품 band inference에는 calibration 적용 여부를 `calibration_applied=false`로 취급해야 한다.
- 1D product history는 `data/parquet/product_prediction_history_1D.parquet` 기준이고, 기존 `/predictions/history`는 제품 차트 history로 쓰지 않는 계약을 유지해야 한다.

## 2. 현재 제품 4슬롯 상태표

| 슬롯 | 현재 run_id | 역할 | gate 계약 | 상태 | 화면 연결 판단 |
|---|---|---|---|---|---|
| 1D line | `patchtst-1D-efad3c29d803` | `line_model` | `line_gate` | `completed`, checkpoint 존재 | 유지 가능 |
| 1D band | `cnn_lstm-1D-d0c780dee5e8` | `band_model` | `band_gate`, `selected_reason=band_gate_eligible` | `completed`, checkpoint 존재 | 유지 가능, raw band로 설명 |
| 1W line | `patchtst-1W-fe7f05a84c93` | `line_model` | `line_gate` | latest-only 저장 완료, asof `2026-05-01` | 유지 가능 |
| 1W band | 없음 | 없음 | 없음 | 준비 중 | 제품 후보 없음으로 유지 |

1M 정책: AI 예측 비활성, 가격 전용. `frontend/src/components/StockView.tsx:370`, `frontend/src/components/StockView.tsx:789`에서 월간 화면은 가격 전용/준비 상태로 처리된다.

## 3. 하드코딩 run_id Inventory

현재 제품 연결 지점:

| 파일 | 근거 | 의미 |
|---|---:|---|
| `frontend/src/components/StockView.tsx` | 86-88, 798-859, 1259-1283 | 주식 화면 1D/1W 제품 run 선택 |
| `frontend/src/components/TrainingView.tsx` | 79-82, 85-119, 527-530, 1765-1767 | 학습 화면 제품 슬롯과 제품 run 제외 처리 |
| `frontend/src/components/BacktestView.tsx` | 18-19, 613-614 | 백테스트 화면의 1D 제품 run 참조 |
| `scripts/check_demo_readiness.ps1` | 7-8 | 데모 준비성 체크의 1D line/band run |
| `scripts/cp99_1d_product_loop_thin_upload.py` | 47-50 | 1D thin upload 리허설 |
| `scripts/cp101_eodhd_off_1d_operation_rehearsal.py` | 40-43 | EODHD-off 1D 리허설 |
| `scripts/cp116_eodhd_off_final_product_loop_rehearsal.py` | 15-16 | EODHD-off 최종 리허설 |
| `scripts/cp128_product_rolling_prediction_history_replay.py` | 20-23 | 1D rolling history replay |
| `scripts/cp140_1w_line_latest_thin_upload.py` | 17, 21 | 1W line latest-only 저장 |
| `data/parquet/product_prediction_history_1D.manifest.json` | manifest 필드 | 1D history line/band run 고정 |

실험/분석 스크립트에도 과거 제품 run id가 남아 있다. 예: `ai/cp106_band_risk_strategy_grid.py`, `ai/cp107_line_trend_strategy_grid.py`, `ai/cp108_lens_indicator_interpretation_study.py`, `ai/cp109_line_band_balance_strategy_grid.py`, `ai/cp110_cash_aware_indicator_balance_strategy_grid.py`. 이들은 제품 화면 연결 지점은 아니지만, 새 run 교체 후 비교 기준이 섞일 수 있으므로 문서상 legacy 분석 스크립트로 분류해야 한다.

## 4. 제품 후보 인정 기준표

| 항목 | LM 후보 인정 | BM 후보 인정 | 제품 후보 제외 |
|---|---|---|---|
| `status` | `completed` | `completed` | `failed_quality_gate`, `failed_nan` |
| `role` | `line_model` | `band_model` | role 누락/불일치, `composite_model` |
| `checkpoint_selection` | `line_gate` | `band_gate` | `val_total` 기본값, smoke 임시값 |
| `selected_reason` | `line_gate_eligible` 권장 | `band_gate_eligible` 권장 | gate fallback/failed reason |
| checkpoint | 경로 존재 및 로드 가능 | 경로 존재 및 로드 가능 | checkpoint 누락/로드 실패 |
| feature 계약 | `feature_version`, `feature_set`, `source_data_hash` 기록 | 동일 | 기록 누락 또는 source/hash 불명확 |
| 저장 계약 | `product_latest_only` | `product_latest_only` | bulk evaluation save, legacy composite save |
| calibration | line display 보정과 학습 성능 구분 | `calibration_applied` 명시 필요 | calibrated 지표를 raw product 성능처럼 기록 |

추가 차단 기준:

- product storage contract가 없는 run은 화면 연결 금지.
- `--save`가 bulk 저장 경로를 타면 제품 저장 금지. 제품 저장은 `save_product_latest_predictions()` 또는 동등한 latest-only helper만 허용.
- `source_data_hash`, `feature_version`, `feature_set`이 문서/DB/config 중 어디에도 없으면 제품 후보 저장 금지.
- line-only row가 schema 때문에 lower/upper를 채웠더라도 meta로 `band_saved=false` 또는 `band_fields_policy=schema_required_degenerate_equal_to_line`를 남기지 않으면 band로 해석 금지.

## 5. 저장 Guard 확인

`ai/storage.py`는 `PRODUCT_LATEST_ALLOWED_LAYERS = {"line", "band"}`와 `STORAGE_CONTRACT_PRODUCT_LATEST_ONLY = "product_latest_only"`를 정의한다. `_validate_product_latest_predictions()`는 single `asof_date`, row 수 제한, composite 차단, layer line/band 제한, duplicate key 차단을 수행한다. `save_product_latest_predictions()`는 저장 전 validation을 거친 뒤 prediction meta에 `storage_contract=product_latest_only`를 붙인다.

`ai/inference_save_guard.py`는 기본 `--save` bulk 저장을 차단한다. `--allow-bulk-evaluation-save`가 없으면 evaluation bulk 저장이 실패하고, product run이 bulk 저장하려 하면 `--save-product-latest-only`를 요구한다. `ai/inference.py:510-513`은 `product_latest_only`일 때 `save_product_latest_predictions()`만 사용하고, 명시적 bulk flag일 때만 `evaluation_bulk`를 사용한다.

판단: latest-only guard는 존재한다. 다만 제품 후보 registry가 없기 때문에 어떤 run이 product run인지 판단하는 메타가 불완전하면 guard가 우회될 수 있다. 다음 full run 저장 전 product candidate annotation을 강제해야 한다.

## 6. Calibration 계약

`ai/band_calibration.py`에는 scalar/conformal calibration 함수가 있고, CP125 1W BM 문서에는 raw 지표와 scalar calibrated 지표가 함께 기록되어 있다. 그러나 `ai/inference.py`에서 확인되는 제품 inference 후처리는 `apply_band_postprocess()`이며, CP125의 `lower_scale`/`upper_scale` 같은 calibration parameter를 제품 저장 경로에서 자동 로드/적용하는 근거는 확인되지 않았다.

따라서 현재 제품 band 슬롯의 성능은 다음처럼 분리해 말해야 한다.

| 구분 | 의미 | 제품 성능으로 말해도 되는가 |
|---|---|---|
| raw band 성능 | checkpoint가 직접 낸 lower/upper | 가능 |
| calibration 참고 성능 | 리포트/JSON에서 후처리 실험한 값 | 제품 저장 경로에 적용 전이면 참고만 가능 |
| product stored band 성능 | latest-only로 저장/조회되는 실제 값 | calibration 적용 여부 명시 필요 |

현재 판단: `calibration_applied=false`. 1W BM 후보 registry에는 calibration 항목이 있어도 1W band 제품 슬롯은 아직 없음/준비 중이다. 새 BM 후보 저장 전에는 `calibration_applied`, `calibration_method`, `calibration_params_ref`를 registry 또는 meta에 명시해야 한다.

## 7. Product History 계약

1D product history는 `data/parquet/product_prediction_history_1D.parquet`와 `data/parquet/product_prediction_history_1D.manifest.json`이 기준이다. manifest 기준:

- line run: `patchtst-1D-efad3c29d803`
- band run: `cnn_lstm-1D-d0c780dee5e8`
- asof range: `2025-05-05` ~ `2026-05-04`
- display horizon: `5`
- row count: `2510`
- ticker count: `5`
- source: `rolling_replay`

`backend/app/services/product_prediction_history_svc.py`는 local parquet만 읽어 product history를 구성하고, `backend/app/routers/v1/stocks.py:112`에서 `/api/v1/stocks/{ticker}/predictions/product-history`를 제공한다. 반면 `backend/app/routers/v1/stocks.py:140`의 `/predictions/history`는 legacy/evaluation 조회로 남아 있으며 제품 차트 history로 쓰면 안 된다.

`frontend/src/components/Chart.tsx:272-299`는 rolling history를 `asof_date` 기준으로 그리고, `forecast_dates` 기반 future forecast와 분리한다. 1W는 아직 product history가 없고 latest forecast만 표시한다. 새 1D product run으로 교체하면 `product_prediction_history_1D.parquet`와 manifest를 반드시 재생성해야 한다.

## 8. Schema/DB 저장 계약

`predictions` 테이블은 line-only row도 `lower_band_series`, `upper_band_series`를 채워야 하는 구조다. CP140은 이를 숨기지 않고 `line_series`와 동일한 퇴화 구간을 저장하며 `meta.band_saved_in_cp140=False`, `meta.band_fields_policy=schema_required_degenerate_equal_to_line`를 남긴다. `frontend/src/components/StockView.tsx:550-560`은 이 meta를 보고 line-only row를 실제 band로 보지 않는다.

판단: 단기 guard는 유지되고 있다. 장기적으로는 line prediction과 band prediction schema를 분리하거나 lower/upper nullable/role-aware로 바꾸는 backlog가 필요하다.

## 9. 현재 화면 유지 가능 항목

- 1D line: 유지 가능. `line_gate`, `line_model`, checkpoint 존재, product history manifest와 일치.
- 1D band: 유지 가능. `band_gate`, `band_model`, checkpoint 존재. 단 calibration 적용 제품으로 말하지 말 것.
- 1W line: 유지 가능. CP140 latest-only 저장에서 predictions 97건, evaluations 0건, composite 0건, `storage_contract=product_latest_only` 확인.
- 1W band: 제품 후보 없음/검증 중 유지.
- 1M: 가격 전용 유지.

## 10. 다음 LM/BM full run 전에 반드시 지킬 것

1. LM 저장은 `--checkpoint-selection line_gate`로만 제품 후보가 된다.
2. BM 저장은 `--checkpoint-selection band_gate`로만 제품 후보가 된다.
3. `val_total` 기본 저장 run은 제품 후보가 아니다.
4. `role`, `checkpoint_selection`, `selected_reason`, `gate_failed`, `line_gate_pass`, `band_gate_pass`, `feature_version`, `source_data_hash`, `feature_set`, `checkpoint_path`를 한 곳에 남겨야 한다.
5. 제품 저장은 `--save-product-latest-only` 또는 `save_product_latest_predictions()`만 사용한다.
6. bulk evaluation save는 명시적 실험/평가 artifact로만 허용하고 제품 저장과 섞지 않는다.
7. band calibration을 제품에 쓰려면 inference 저장 경로에서 실제 적용하고 meta에 `calibration_applied=true`를 남긴다.

## 11. 모델 저장 후 사람이 바꿔야 하는 연결 지점

새 1D/1W 제품 run 저장 후 수동 변경이 필요한 지점:

- `frontend/src/components/StockView.tsx`: 제품 화면 run id 상수와 1W band 상태.
- `frontend/src/components/TrainingView.tsx`: `PRODUCT_RUN_IDS`, `PRODUCT_SLOTS`, `getRunRole`.
- `frontend/src/components/BacktestView.tsx`: 1D line/band backtest 기준 run.
- `scripts/check_demo_readiness.ps1`: 데모 readiness 기준 run.
- product thin upload/rehearsal scripts: CP99/CP101/CP116/CP140 계열.
- `scripts/cp128_product_rolling_prediction_history_replay.py`: 1D history replay 대상 run.
- `data/parquet/product_prediction_history_1D.manifest.json`: line/band run id, source hash, date range.

## 12. 나중에 자동화할 항목

- `product_model_registry.json` 또는 DB thin table로 1D line, 1D band, 1W line, 1W band 슬롯을 중앙 관리.
- 제품 후보 승인 CLI: gate/role/checkpoint/source hash/storage contract/calibration policy를 한 번에 검증.
- `predictions` schema 개선: line-only와 band-only 저장을 role-aware로 분리.
- calibration artifact registry: calibration params를 제품 inference가 재사용하고 meta에 적용 여부를 남김.
- product history replay 자동화: 제품 run 교체 시 parquet와 manifest 자동 재생성.
- `/predictions/history` legacy endpoint에 제품 차트 사용 금지 표시 또는 endpoint 이름 정리.

## 13. 다음 CP 추천

1. `product_model_registry` 최소 JSON/loader 추가.
2. BM calibration 적용 여부를 제품 inference meta에 명시하는 guard 추가.
3. product candidate annotation 검증 테스트 추가.
4. 새 500티커 LM/BM full run 저장 전 `line_gate`/`band_gate` 저장 명령 템플릿 고정.

## 14. 읽기 전용 명령 목록

- `Select-String`으로 제품 run id, storage guard, product history endpoint, CP140 meta를 검색했다.
- JSON 문서와 manifest를 읽어 1D/1W 제품 run의 gate, role, source hash, checkpoint 존재 여부를 확인했다.
- checkpoint 파일 존재 여부를 `Test-Path`와 Python `Path.exists()`로 확인했다.
- DB write, inference 저장, 모델 학습, Supabase raw read/write, 프론트 수정은 수행하지 않았다.

## 15. 최종 구분

지금 당장 제품 화면에 유지해도 되는 것:

- 1D line `patchtst-1D-efad3c29d803`
- 1D band `cnn_lstm-1D-d0c780dee5e8`, 단 raw band로 설명
- 1W line `patchtst-1W-fe7f05a84c93`
- 1W band 준비 중, 1M 가격 전용

다음 LM/BM full run 전에 반드시 지켜야 할 것:

- LM은 `line_gate`, BM은 `band_gate`만 제품 후보.
- `val_total`/smoke/failed run 자동 승격 금지.
- storage contract는 `product_latest_only`.
- source hash와 feature contract 기록 필수.
- calibration 성능과 product stored 성능 분리.

모델 저장 후 사람이 바꿔야 하는 연결 지점:

- `StockView.tsx`, `TrainingView.tsx`, `BacktestView.tsx`
- demo/readiness/thin upload/replay scripts
- `product_prediction_history_1D.parquet`와 manifest

나중에 schema/registry로 자동화할 항목:

- 제품 4슬롯 중앙 registry
- product candidate 승인 CLI/test
- line/band role-aware prediction schema
- calibration artifact 적용/기록 계약
- product history replay 자동 갱신
