# CP68-LM h20 conservative line rescue

## 1. 원칙 확인
- 이번 CP는 line_model 전용 post-hoc calibration이다.
- 새 학습은 하지 않았고 CP67 h20_full_features 100티커 checkpoint 예측만 재사용했다.
- band 모델 실험, composite/overlay, line_inside_band 평가는 사용하지 않았다.
- DB 쓰기, save-run, W&B, full 473티커, UI/backend 수정, feature contract 변경은 하지 않았다.

## 2. 입력과 cache 확인
| 항목 | 값 | 판정 |
| --- | --- | --- |
| checkpoint | ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-8a5e01c0f891.pt | 사용 |
| source_data_hash | f7c7b101 | PASS |
| feature_version | v3_adjusted_ohlc | PASS |
| MODEL_FEATURE_COLUMNS | 36 | PASS |
| atr_ratio in model | False | PASS |
| eligible ticker | 93 | limit 100 scope |
| feature cache created | False | 기존 cache 재사용 |
| feature NaN/Inf | 0 | PASS |
| target NaN/Inf | 0 | PASS |

## 3. validation fit 정책
| 정책 | 방식 | offset | bucket offsets | val false_safe_tail | target |
| --- | --- | --- | --- | --- | --- |
| raw_line | none | 0.0 |  |  | None |
| global_downshift | validation_global_offset | 0.01572489086896145 |  | 0.3000 | True |
| horizon_bucket_downshift | validation_bucket_offsets |  | {'h1_h5': 0.0113019, 'h6_h10': 0.01381167, 'h11_h20': 0.01482145} |  | True |
| volatility_scaled_downshift | validation_volatility_scaled_offset | 0.015587985981321604 |  | 0.3000 | True |

offset은 validation split에서만 산정했고 test 지표를 본 뒤 고르지 않았다.

## 4. test line 결과
| 정책 | IC | spread | false_safe_tail | false_safe_severe | severe_recall | bias | upside_sacrifice | fee_ret | fee_sharpe | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_line | 0.0667 | 0.0280 | 0.3918 | 0.3810 | 0.6190 | -0.0205 | 0.1104 | 12115.2083 | 0.4219 | FAIL |
| global_downshift | 0.0667 | 0.0280 | 0.2756 | 0.2742 | 0.7258 | -0.0362 | 0.1261 | 12115.2083 | 0.4219 | PASS |
| horizon_bucket_downshift | 0.0667 | 0.0280 | 0.2857 | 0.2829 | 0.7171 | -0.0342 | 0.1247 | 12115.2083 | 0.4219 | PASS |
| volatility_scaled_downshift | 0.0632 | 0.0256 | 0.2576 | 0.2526 | 0.7474 | -0.0398 | 0.1316 | 4958.3154 | 0.3756 | PASS |

## 5. bucket별 test 결과
| 정책 | bucket | IC | spread | false_safe_tail | severe_recall | fee_ret |
| --- | --- | --- | --- | --- | --- | --- |
| raw_line | h1_h5 | 0.0477 | 0.0037 | 0.3583 | 0.6335 | 2.1689 |
| raw_line | h6_h10 | 0.0526 | 0.0100 | 0.3862 | 0.6192 | 24.5301 |
| raw_line | h11_h20 | 0.0633 | 0.0203 | 0.3829 | 0.6295 | 951.9495 |
| global_downshift | h1_h5 | 0.0477 | 0.0037 | 0.2397 | 0.7241 | 2.1689 |
| global_downshift | h6_h10 | 0.0526 | 0.0100 | 0.2606 | 0.7325 | 24.5301 |
| global_downshift | h11_h20 | 0.0633 | 0.0203 | 0.2591 | 0.7356 | 951.9495 |
| horizon_bucket_downshift | h1_h5 | 0.0477 | 0.0037 | 0.2747 | 0.7044 | 2.1689 |
| horizon_bucket_downshift | h6_h10 | 0.0526 | 0.0100 | 0.2760 | 0.7155 | 24.5301 |
| horizon_bucket_downshift | h11_h20 | 0.0633 | 0.0203 | 0.2648 | 0.7309 | 951.9495 |
| volatility_scaled_downshift | h1_h5 | 0.0467 | 0.0027 | 0.2270 | 0.7626 | 1.2141 |
| volatility_scaled_downshift | h6_h10 | 0.0505 | 0.0088 | 0.2392 | 0.7632 | 15.2166 |
| volatility_scaled_downshift | h11_h20 | 0.0603 | 0.0186 | 0.2403 | 0.7612 | 502.7919 |

## 6. 핵심 질문 답변
- h20 raw expected line은 랭킹용으로는 쓸 만하다. raw test IC=0.0667, spread=0.0280로 양수다.
- h20 conservative line은 best 정책 `horizon_bucket_downshift` 기준 false_safe_tail=0.2857, severe_recall=0.7171이다.
- IC/spread 희생은 best 정책 기준 delta IC=0.0000, delta spread=0.0000이다.

## 7. 제품 표시 판단
- 최종 분류: PASS
- 표시 제안: h20 conservative line을 기본 OFF / 사용자 선택형 중기 보수 판단선 후보로 유지한다.
- h20 raw line과 h20 conservative line은 분리해서 기록한다.
- h20 raw line이 IC/spread가 좋아도 false-safe가 높으면 위험선으로 쓰지 않는다.
- h5는 단기 line 진한 실선 역할을 유지한다.
- h20 conservative line이 PASS이면 기본 OFF / 사용자 선택형 중기 보수 판단선으로만 둔다.

## 8. post-hoc 이후 학습 loss 후보
- h20 전용 asymmetric line loss
- overprediction penalty 강화
- severe downside sample weighting
- direction/risk auxiliary head
- h20 전용 feature set 재검토

## 9. 다음 LM 추천
CP69-LM에서는 같은 post-hoc 정책을 h20 no_fundamentals와 CP49 h20 dense/risk-only 후보에 적용해 보수선 안정성을 비교한다.

## 10. 검증
- `.venv\Scripts\python.exe -m py_compile ai\cp68_lm_h20_conservative_line_rescue.py ai\tests\test_cp68_conservative_calibration.py`: 통과
- `python -m json.tool docs\cp68_lm_h20_conservative_line_rescue_metrics.json`: 통과
- `.venv\Scripts\python.exe -m unittest ai.tests.test_cp68_conservative_calibration ai.tests.test_feature_set_selection ai.tests.test_checkpoint_selection ai.tests.test_metric_definition_contract ai.tests.test_splits`: 30개 통과
- 마지막 확인 기준 잔여 `python/pythonw` 학습 프로세스 없음
