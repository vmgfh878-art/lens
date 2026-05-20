# CP204-PG v1 Production Model Set Save-Run

1W line은 deferred다. 이번 save-run 대상이 아니며, 프론트는 현재 동작 그대로 유지한다.

## 최종 판정: `WARN_CP204_V1_PARTIAL_READY`

## Target Lock

| slot_name | source_cp | candidate | role | timeframe | source_path | status | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1D Line | CP175 | TiDE beta=5 frozen | line_model | 1D | C:\Users\user\lens\data\tmp\cp194r2_safe_line_sets\line_b_test.parquet | LOCKED | 사용자 지정 line_b_test.parquet를 우선 사용한다. 1D line frozen source 여부는 parquet meta columns로 확인한다. |
| 1D Band | CP153 | tide_s60_q15_param | band_model | 1D | C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_latest_predictions.json | LOCKED | CP202 target lock의 1D primary band artifact를 사용한다. |
| 1W Band | CP178 | walk_forward_lower_calibration on tide_s60_q10_q90_param | band_model | 1W | C:\Users\user\lens\docs\cp178_wflock_1w_band_walk_forward_lower_metrics.json | LOCKED_BUT_PREDICTION_SOURCE_UNRESOLVED | CP202 target lock의 1W artifact는 metrics JSON이다. row-level prediction payload는 아직 확인되지 않았다. |
| 1W Line |  | Deferred | line_model | 1W |  | DEFERRED_NOT_SAVE_RUN_TARGET | 이번 save-run 대상이 아니며 프론트는 현재 동작 그대로 유지한다. |

## Preflight

| slot_name | source_cp | source_path | source_exists | role_compatibility | feature_column_order | manifest_meta_parse | timeframe_horizon_role | calibration_resolve | frozen_source_usable | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D Line | CP175 | C:\Users\user\lens\data\tmp\cp194r2_safe_line_sets\line_b_test.parquet | True | PASS | PASS | PASS | PASS | N/A | PASS | line_id=['line_b_cp175_beta5'], line_model_id=['line_b_cp175_beta5_frozen_eval'] |
| 1D Band | CP153 | C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_latest_predictions.json | True | PASS | PASS | PASS | PASS | PASS | PASS | records=474, run_id=tide-1D-ea54dcae654d, calibration=lower_focused |
| 1W Band | CP178 | C:\Users\user\lens\docs\cp178_wflock_1w_band_walk_forward_lower_metrics.json | True | PASS | PASS | PASS | PASS | PASS | FAIL | candidate=tide_s60_q10_q90_param, method=walk_forward_lower_calibration, horizon=4, row_level_records=False / prediction parquet 생성에는 row-level payload가 필요해 추정 중단 |

## Save-Run Summary

| slot_name | source_cp | source_path | source_data_hash | timeframe | role | model_id | row_count | ticker_count | date_range | calibration_used | drift_waiver_note | output_path | calibration_method | calibration_params | status | block_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D Line | CP175 | C:\Users\user\lens\data\tmp\cp194r2_safe_line_sets\line_b_test.parquet | ec39d198380f42ce | 1D | line_model | line_b_cp175_beta5_frozen_eval | 177095 | 471 | ['2024-10-30', '2026-05-01'] | False | CP204 local frozen import package. 새 학습, 새 calibration, checkpoint 재선택, DB/Supabase write 없음. 배포 업로드는 별도 승인 전까지 보류. | C:\Users\user\lens\docs\cp204_save_run\01_1d_line_cp175\predictions.parquet |  |  |  |  |
| 1D Band | CP153 | C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_latest_predictions.json | 90666b44cbfb8e5c | 1D | band_model | tide-1D-ea54dcae654d | 474 | 474 | ['2026-05-08', '2026-05-08'] | True | CP204 local frozen import package. 새 학습, 새 calibration, checkpoint 재선택, DB/Supabase write 없음. 배포 업로드는 별도 승인 전까지 보류. | C:\Users\user\lens\docs\cp204_save_run\02_1d_band_cp153\predictions.parquet | lower_focused | {'lower_scale': 1.05, 'upper_scale': 1.0} |  |  |
| 1W Band | CP178 | C:\Users\user\lens\docs\cp178_wflock_1w_band_walk_forward_lower_metrics.json | 90666b44cbfb8e5c | 1W | band_model | tide_s60_q10_q90_param | 0 | 451 |  | True | CP204 local frozen import package. 새 학습, 새 calibration, checkpoint 재선택, DB/Supabase write 없음. 배포 업로드는 별도 승인 전까지 보류. |  | walk_forward_lower_calibration | {'source': 'walk-forward fold metrics only'} | BLOCKED_NO_ROW_LEVEL_PREDICTION_PAYLOAD | CP178 확정 artifact는 metrics JSON이며 prediction records/parquet가 없다. 추정 생성 금지 원칙에 따라 1W band save-run은 수행하지 않았다. |

## Validation

| slot_name | source_cp | parquet_read | duplicate_key_count | non_finite_count | expected_timeframe | manifest_json_parse | report_utf8_read | schema_check | status | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D Line | CP175 | PASS | 0 | 0 | PASS | PASS | PASS | PASS | PASS |  |
| 1D Band | CP153 | PASS | 0 | 0 | PASS | PASS | PASS | PASS | PASS |  |
| 1W Band | CP178 | N/A |  |  | N/A | PASS | PASS | N/A | BLOCKED | CP178 확정 artifact는 metrics JSON이며 prediction records/parquet가 없다. 추정 생성 금지 원칙에 따라 1W band save-run은 수행하지 않았다. |

## 금지 항목 준수

- 새 학습 없음
- 새 calibration 없음
- checkpoint 재선택 없음
- DB/Supabase write 없음
- save-run 결과 업로드 없음
- 프론트/README 수정 없음

## CP204.2 Drift Acceptance Update

- verdict: `PASS_CP204_V1_READY`
- drift waiver: 1W Band predictions use CP178 frozen 9-checkpoint ensemble applied to current CP163 calendar split. Row count (416,724) differs from CP178 frozen metrics (411,312) due to fold_3 test window extension. Same checkpoints + same calibration params used; difference is split definition only. Drift waiver accepted for v1 product (Option A in CP204.2). Exact CP178 metric reproduction requires re-smoke on current split (deferred to v1.1).
- 1W Band parquet: `C:\Users\user\lens\data\artifacts\cp204_v1_import_package\cp204_1w_band_payload.parquet`
- 1W Line은 계속 deferred다.
