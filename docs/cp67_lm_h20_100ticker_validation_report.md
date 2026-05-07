# CP67-LM 1D h20 100티커 재검증

## 1. 원칙 확인
- 이번 CP는 PatchTST line_model만 실행했다.
- band 모델, composite/overlay, line_inside_band 관련 실험과 평가는 수행하지 않았다.
- DB 쓰기, save-run, W&B, full 473티커, UI/backend API 수정, feature contract 변경은 하지 않았다.
- `atr_ratio`는 모델 feature로 승격하지 않았다.

## 2. cache gate
| 항목 | 값 | 판정 |
| --- | --- | --- |
| current hash | f7c7b101 | PASS |
| CP66 hash | f7c7b101 |  |
| feature_version | v3_adjusted_ohlc | PASS |
| MODEL_FEATURE_COLUMNS | 36 | PASS |
| atr_ratio in model | False | PASS |
| eligible ticker | 93 | PASS |
| feature NaN/Inf | 0 | PASS |
| target NaN/Inf | 0 | PASS |
| ratio p99 sanity | True | PASS |

## 3. 100티커 line 결과
| name | feature_set | status | n | ic | ic_ir | ic_t | spread | spread_ir | spread_t | dir | false_safe_tail | false_safe_severe | severe_recall | capture | bias | sacrifice | mae | smape | fee_ret | fee_sharpe | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h20_full_features_post_backfill_100 | full_features | completed | 36 | 0.0667 | 0.3650 | 7.0500 | 0.0280 | 0.4258 | 8.2237 | 0.5055 | 0.3918 | 0.3810 | 0.6190 | 0.2395 | -0.0205 | 0.1104 | 0.0657 | 1.4632 | 12115.2083 | 0.4219 | fail |

## 4. horizon bucket 지표
| name | bucket | ic | spread | false_safe_tail | severe_recall | fee_ret |
| --- | --- | --- | --- | --- | --- | --- |
| h20_full_features_post_backfill_100 | h1_h5 | 0.0477 | 0.0037 | 0.3583 | 0.6335 | 2.1689 |
| h20_full_features_post_backfill_100 | h6_h10 | 0.0526 | 0.0100 | 0.3862 | 0.6192 | 24.5301 |
| h20_full_features_post_backfill_100 | h11_h20 | 0.0633 | 0.0203 | 0.3829 | 0.6295 | 951.9495 |

## 5. 비교 기준
| source | name | ic | spread | false_safe_tail | severe_recall |
| --- | --- | --- | --- | --- | --- |
| CP66 50 | h20_full_features_post_backfill_seq252_p32_s16 | 0.0230 | 0.0067 | 0.2895 | 0.7120 |
| CP66 50 | h20_no_fundamentals_post_backfill_seq252_p32_s16 | 0.0214 | 0.0136 | 0.3361 | 0.6647 |
| CP65 | technical_only:h20_technical_only_seq252_p32_s16 | 0.0137 | 0.0062 | 0.3375 | 0.6645 |
| CP65 | no_fundamentals:h20_no_fundamentals_seq252_p32_s16 | 0.0230 | 0.0147 | 0.3015 | 0.6993 |
| CP65 | price_volatility_volume:h20_price_volatility_volume_seq252_p32_s16 | 0.0137 | 0.0062 | 0.3375 | 0.6645 |
| CP65 | full_features:h20_longer_context_seq252_p32_s16 | 0.0199 | 0.0072 | 0.3149 | 0.6868 |
| CP49 | h20_baseline_seq252_p16_s8 | -0.0068 | 0.0022 | 0.2676 | 0.7321 |
| CP49 | h20_longer_context_seq252_p32_s16 | 0.0199 | 0.0072 | 0.3149 | 0.6868 |
| CP49 | h20_dense_overlap_seq252_p16_s4 | 0.0138 | 0.0018 | 0.1091 | 0.8899 |
| CP49 | h20_baseline_seq504_p16_s8 | -0.0521 | -0.0264 | 0.0388 | 0.9604 |
| baseline | historical_mean_line_w60 | 0.0211 | 0.0031 | 0.5541 | 0.4495 |
| baseline | reversal_line_horizon | 0.0120 | 0.0046 | 0.4860 | 0.5085 |
| baseline | random_or_shuffled_score | 0.0213 | 0.0019 | 0.5113 | 0.4989 |

## 6. 실행 스킵
| name | reason |
| --- | --- |
| h20_no_fundamentals_post_backfill_100 | A full_features 100티커가 fail이라 지시대로 B를 실행하지 않았다. |

## 7. 제품 표시 판단
- h5: 단기 line, 진한 실선 유지.
- h20: Phase 1.5 연구 후보 보류.
- 기본 ON 여부: NO.
- 판단 근거: fail 조건이 있어 제품 표시를 금지한다.

## 8. 선택 실험 C
- 실행 여부: not_run
- 예상 시간: None
- 사유: A가 product_aux_pass가 아니므로 선택 5epoch 장기 실행 조건을 만족하지 않았다.

## 9. 다음 CP 추천
- 100티커 h20 full_features는 제품 기본 ON 후보가 아니다. CP66 50티커 생존 결과는 표본 확대에서 재현되지 않았다.
- h20은 Phase 1.5 연구 후보로 보류하고, 제품에는 h5 단기 line을 기본 진한 실선으로 유지한다.
- h20을 계속 보려면 100티커에서 false-safe를 낮추는 재학습/seed 안정성 또는 후보 구조를 별도 CP로 분리한다.

## 10. 검증
- `.venv\Scripts\python.exe -m py_compile ai\cp67_lm_h20_100ticker_validation.py`: 통과
- `python -m json.tool docs\cp67_lm_h20_100ticker_validation_metrics.json`: 통과
- `.venv\Scripts\python.exe -m unittest ai.tests.test_feature_set_selection ai.tests.test_checkpoint_selection ai.tests.test_metric_definition_contract ai.tests.test_splits`: 27개 통과
- 마지막 확인 기준 잔여 `python/pythonw` 학습 프로세스 없음
