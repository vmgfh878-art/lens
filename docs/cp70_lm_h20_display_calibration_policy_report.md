# CP70-LM h20 display calibration policy 비교

## 1. 실행 원칙
- 새 학습은 하지 않았다.
- CP67 checkpoint 기반 forward-only / metrics-only post-hoc 평가만 수행했다.
- validation에서 offset을 fit하고 test에는 고정 적용했다.
- DB 쓰기, save-run, W&B, full 473티커, UI/backend 수정, band/composite 실험은 하지 않았다.
- CP68 display calibration을 `trained_conservative_line`으로 부르지 않고 `display_calibrated_line`으로 분리했다.

## 2. GPU 사용
| 항목 | 값 | 비고 |
| --- | --- | --- |
| requested_device | cpu |  |
| resolved_device | cpu | CPU-only |
| gpu_used | False |  |
| elapsed_seconds | 424.8113 |  |
| peak_vram_mb |  | 해당 없음 |

이번 CP는 CPU-only로 실행했다. GPU는 사용하지 않았다.

## 3. 데이터/cache 주의
| 항목 | 값 | 판정 |
| --- | --- | --- |
| current source hash | 3ac43945 | MISMATCH |
| CP67 source hash | f7c7b101 |  |
| feature_version | v3_adjusted_ohlc | PASS |
| eligible ticker | 93 |  |
| feature/target NaN/Inf | 0 / 0 | PASS |
| ratio sanity | True | PASS |
| feature cache created | 실행 로그상 재생성 메시지 출력 |  |

CP70 실행 시 current source hash가 CP67 hash와 달랐다. 따라서 이 결과는 `CP67 checkpoint + 현재 100티커 cache/data` 기준의 정책 안정성 재확인이다. 정확한 CP68 동일-hash 재현은 아니며, 이 차이는 제품 판단의 잔여 리스크로 기록한다.

## 4. validation fit 정책
| 정책 | line_type | 방식 | offset | bucket offsets | val false_safe_tail | target |
| --- | --- | --- | --- | --- | --- | --- |
| raw_model_line | raw_model_line / trained_conservative_line | none | 0.0 |  |  | None |
| global_downshift | display_calibrated_line | validation_global_offset | 0.01578281541052122 |  | 0.3000 | True |
| horizon_bucket_downshift | display_calibrated_line | validation_bucket_offsets |  | {'h1_h5': 0.01147843, 'h6_h10': 0.01394746, 'h11_h20': 0.01498582} |  | True |
| volatility_scaled_downshift | display_calibrated_line | validation_volatility_scaled_offset | 0.01563751045737436 |  | 0.3000 | True |

## 5. 100티커 test line 지표
| 정책 | line_type | IC | IC IR | IC t | spread | spread IR | spread t | false_safe_tail | false_safe_severe | severe_recall | bias | upside_sacrifice | fee_ret | fee_sharpe | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_model_line | raw_model_line / trained_conservative_line | 0.0676 | 0.3674 | 7.0955 | 0.0284 | 0.4309 | 8.3227 | 0.3911 | 0.3804 | 0.6196 | -0.0207 | 0.1107 | 14304.6514 | 0.4271 | fail |
| global_downshift | display_calibrated_line | 0.0676 | 0.3674 | 7.0955 | 0.0284 | 0.4309 | 8.3227 | 0.2744 | 0.2731 | 0.7269 | -0.0364 | 0.1265 | 14304.6514 | 0.4271 | default_off_candidate |
| horizon_bucket_downshift | display_calibrated_line | 0.0676 | 0.3674 | 7.0955 | 0.0284 | 0.4309 | 8.3227 | 0.2841 | 0.2814 | 0.7186 | -0.0345 | 0.1251 | 14304.6514 | 0.4271 | default_off_candidate |
| volatility_scaled_downshift | display_calibrated_line | 0.0640 | 0.3401 | 6.5676 | 0.0259 | 0.3819 | 7.3756 | 0.2566 | 0.2516 | 0.7484 | -0.0401 | 0.1320 | 5423.8341 | 0.3779 | default_off_candidate |

## 6. bucket별 test 지표
| 정책 | bucket | IC | spread | false_safe_tail | severe_recall | upside_sacrifice |
| --- | --- | --- | --- | --- | --- | --- |
| raw_model_line | h1_h5 | 0.0486 | 0.0037 | 0.3571 | 0.6339 | 0.0641 |
| raw_model_line | h6_h10 | 0.0530 | 0.0102 | 0.3850 | 0.6198 | 0.0950 |
| raw_model_line | h11_h20 | 0.0637 | 0.0205 | 0.3817 | 0.6305 | 0.1265 |
| global_downshift | h1_h5 | 0.0486 | 0.0037 | 0.2378 | 0.7254 | 0.0798 |
| global_downshift | h6_h10 | 0.0530 | 0.0102 | 0.2597 | 0.7333 | 0.1107 |
| global_downshift | h11_h20 | 0.0637 | 0.0205 | 0.2584 | 0.7366 | 0.1423 |
| horizon_bucket_downshift | h1_h5 | 0.0486 | 0.0037 | 0.2726 | 0.7057 | 0.0755 |
| horizon_bucket_downshift | h6_h10 | 0.0530 | 0.0102 | 0.2746 | 0.7168 | 0.1089 |
| horizon_bucket_downshift | h11_h20 | 0.0637 | 0.0205 | 0.2630 | 0.7328 | 0.1415 |
| volatility_scaled_downshift | h1_h5 | 0.0476 | 0.0029 | 0.2251 | 0.7628 | 0.0854 |
| volatility_scaled_downshift | h6_h10 | 0.0510 | 0.0089 | 0.2383 | 0.7640 | 0.1166 |
| volatility_scaled_downshift | h11_h20 | 0.0607 | 0.0188 | 0.2395 | 0.7628 | 0.1481 |

## 7. 정책 비교
- `raw_model_line`은 alpha=1 beta=2 loss로 학습된 `trained_conservative_line` 원출력이지만 false_safe_tail이 높아 표시 후보가 아니다.
- `global_downshift`는 단순하고 안정적이며 IC/spread/fee를 보존했다.
- `horizon_bucket_downshift`는 horizon별 오차 차이를 반영하면서 IC/spread/fee를 보존했고 h11_h20 false_safe 기준도 통과했다.
- `volatility_scaled_downshift`는 false_safe를 가장 낮췄지만 IC/spread/fee 희생이 있고, 지표 기반 scale이 validation 분포에 과적합될 가능성이 있다.

## 8. 제품 표시 후보 결정
- 최종 추천: A. h20 display_calibrated_line을 제품 기본 OFF 후보로 유지
- 선택 정책: horizon_bucket_downshift
- 이유: false_safe_tail/severe_recall/h11_h20 기준을 통과하면서 IC/spread/fee 희생이 없고, horizon별 오차 차이를 반영해 해석성이 global_downshift보다 좋다.
- h20 표시선은 제품 기본 ON이 아니라 기본 OFF / 사용자 선택형 중기 참고선으로 둔다.
- h5 단기 line은 기존처럼 제품 기본 예측선 역할을 유지한다.

## 9. 다음 단계
제품 연결 전에는 h20 display_calibrated_line을 기본 OFF 토글 후보로 문서화하고, 별도 학습 CP에서는 beta 3~4 또는 severe downside weighting을 검토한다. 이번 CP에서는 구현하지 않았다.

## 10. 검증
- `.venv\Scripts\python.exe -m py_compile ai\cp70_lm_h20_display_calibration_policy.py ai\tests\test_cp70_display_policy.py`: 통과
- `python -m json.tool docs\cp70_lm_h20_display_calibration_policy_metrics.json`: 통과
- `.venv\Scripts\python.exe -m unittest ai.tests.test_cp70_display_policy ai.tests.test_cp68_conservative_calibration ai.tests.test_losses ai.tests.test_loss`: 13개 통과
- 마지막 확인 기준 잔여 `python/pythonw` 프로세스 없음
