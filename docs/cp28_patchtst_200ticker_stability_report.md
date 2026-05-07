# CP28-R PatchTST 200티커 Coverage-Gate Stability 보고서

## 1. 목표
CP27에서 복구된 `q20-b2`, `q25-b2`, `q15-b2` 후보가 100티커를 넘어 200티커에서도 유지되는지 확인했다. 통과 후보만 full 473티커 후보로 올리는 것이 목적이며, 모델 구조와 loss는 변경하지 않았다.

## 2. 실행 조건
공통 조건은 `patchtst`, `1D`, `seq_len=252`, `epochs=3`, `batch_size=256`, `device=cuda`, `ci_aggregate=target`, `raw_future_return`, `limit_tickers=200`, `patch_len=16`, `patch_stride=8`, `band_mode=direct`, `lambda_band=2.0`, `checkpoint_selection=coverage_gate`이다.

## 3. 데이터 규모
입력 200티커 중 재무 sufficiency gate로 12개가 제외되어 실제 학습 대상은 188티커였다. 샘플 수는 train 313,286개, validation 67,095개, test 67,194개였다.

## 4. 100티커 대비 200티커 결과
| 후보 | 100 selected_epoch | 100 coverage | 100 upper | 100 fee | 200 selected_epoch | 200 coverage | 200 upper | 200 fee | coverage_gate_passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| q20-b2 | 1 | 0.9244 | 0.0539 | 0.7930 | 3 | 0.5984 | 0.2243 | 356.5852 | false |
| q25-b2 | 1 | 0.8897 | 0.0813 | 0.4795 | 3 | 0.4993 | 0.2770 | 347.0895 | false |
| q15-b2 | 2 | 0.8026 | 0.0953 | 0.5457 | 3 | 0.7021 | 0.1663 | 385.7829 | false |

## 5. Validation 상세 지표
| 후보 | selected_epoch | val_total_best_epoch | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction | selected_reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| q20-b2 | 3 | 3 | 0.5984 | 0.1773 | 0.2243 | 0.0938 | 4.3731 | 0.0318 | 0.0559 | 0.0107 | 356.5852 | 0.4867 | 0.3757 | 0.0425 | coverage_gate_failed_fallback_val_total |
| q25-b2 | 3 | 3 | 0.4993 | 0.2238 | 0.2770 | 0.0749 | 3.4933 | 0.0360 | 0.0557 | 0.0107 | 347.0895 | 0.4875 | 0.3793 | 0.0427 | coverage_gate_failed_fallback_val_total |
| q15-b2 | 3 | 3 | 0.7021 | 0.1316 | 0.1663 | 0.1170 | 5.4554 | 0.0267 | 0.0565 | 0.0108 | 385.7829 | 0.4868 | 0.3760 | 0.0425 | coverage_gate_failed_fallback_val_total |

## 6. Test 상세 지표
| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q20-b2 | 0.5907 | 0.1803 | 0.2289 | 0.0994 | 4.0489 | 0.0246 | 0.0206 | 0.0008 | -0.0433 | 0.4938 | 0.3948 | 0.0413 |
| q25-b2 | 0.4908 | 0.2305 | 0.2788 | 0.0794 | 3.2346 | 0.0271 | 0.0207 | 0.0007 | -0.0817 | 0.4943 | 0.3987 | 0.0415 |
| q15-b2 | 0.6947 | 0.1318 | 0.1735 | 0.1241 | 5.0521 | 0.0227 | 0.0207 | 0.0008 | -0.0294 | 0.4940 | 0.3951 | 0.0413 |

## 7. Epoch별 진단
| 후보 | epoch | coverage | upper_breach_rate | avg_band_width | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|---:|
| q20-b2 | 1 | 0.6470 | 0.2411 | 0.1086 | 0.0612 | 0.0101 | 243.9122 |
| q20-b2 | 2 | 0.6017 | 0.1991 | 0.0943 | 0.0543 | 0.0094 | 135.7681 |
| q20-b2 | 3 | 0.5984 | 0.2243 | 0.0938 | 0.0559 | 0.0107 | 356.5852 |
| q25-b2 | 1 | 0.5228 | 0.3166 | 0.0838 | 0.0584 | 0.0093 | 149.3798 |
| q25-b2 | 2 | 0.5126 | 0.2407 | 0.0767 | 0.0561 | 0.0103 | 280.9512 |
| q25-b2 | 3 | 0.4993 | 0.2770 | 0.0749 | 0.0557 | 0.0107 | 347.0895 |
| q15-b2 | 1 | 0.7577 | 0.1689 | 0.1382 | 0.0617 | 0.0103 | 273.7387 |
| q15-b2 | 2 | 0.6996 | 0.1511 | 0.1162 | 0.0544 | 0.0095 | 147.4965 |
| q15-b2 | 3 | 0.7021 | 0.1663 | 0.1170 | 0.0565 | 0.0108 | 385.7829 |

## 8. 속도와 자원
| 후보 | 평균 epoch_seconds | VRAM peak MB | 종료코드 |
|---|---:|---:|---:|
| q20-b2 | 291.02 | 5153.31 | 0 |
| q25-b2 | 290.18 | 5153.31 | 0 |
| q15-b2 | 291.28 | 5153.31 | 0 |

## 9. 판정
q20-b2와 q25-b2는 coverage와 upper breach 기준을 크게 벗어나 탈락했다. q15-b2는 epoch1에서 coverage 0.7577로 하한은 넘겼지만 upper breach 0.1689가 공격형 한계 0.15를 넘었고, 선택 checkpoint도 fallback이었다. 따라서 CP28 기준에서 full 473티커로 올릴 후보는 없다.

## 10. 해석
200티커로 확장하자 투자 지표의 validation 값은 강하게 양수로 보이지만, test fee return은 모두 약한 음수이고 coverage/breach는 명확히 무너졌다. 이는 밴드 calibration이 ticker universe 확장에 안정적으로 전이되지 않는다는 뜻이다. PatchTST를 즉시 폐기할 정도는 아니지만, 현재 preset을 full run으로 보내면 같은 문제가 더 커질 가능성이 높다.

## 11. 결론
CP28은 통과하지 못했다. q20/q25/q15 모두 200티커 coverage gate를 통과하지 못했으므로 full 473티커 실행은 계속 금지한다. 다음 단계는 PatchTST preset을 더 미세 조정하기보다, DLinear/NLinear 같은 단순 baseline을 준비해 현재 PatchTST 밴드 문제가 모델 복잡도 문제인지 비교하는 쪽이 맞다.
