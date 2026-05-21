CP34는 새 모델 추가 CP가 아니라, 예측선 실험과 밴드 실험을 분리하고 TiDE/CNN-LSTM을 밴드 후보로 재평가하는 CP다.

# CP34-M Line/Band 실험 분리 보고서

## 1. 목표
PatchTST direct band 실패 이후 예측선 실험과 밴드 실험의 평가판을 분리했다. 동시에 `v3_adjusted_ohlc` clean feature 기준에서 TiDE와 CNN-LSTM을 밴드 후보로 짧게 재평가했다.

## 2. 평가 체계 분리
예측선 평가는 선 자체의 방향성과 순위 신호를 본다.

| 분류 | 지표 |
|---|---|
| 예측선 | `mae`, `smape`, `direction_accuracy`, `spearman_ic`, `top_k_long_spread`, `top_k_short_spread`, `long_short_spread` |
| 밴드 | `coverage`, `lower_breach_rate`, `upper_breach_rate`, `avg_band_width`, `normalized_band_width`, `band_loss`, `cross_loss`, `coverage_gate_failed`, `selected_epoch` |
| 시그널/백테스트 | `fee_adjusted_return`, `fee_adjusted_sharpe`, `turnover` |

이번 CP에서 시그널/백테스트 지표는 보조 지표로만 사용했다. 규칙 기반 시그널이 아직 단순하므로 fee 음수만으로 밴드 후보를 바로 탈락시키지는 않았다.

## 3. 모델 역할 표
| 모델 | 현재 역할 | 판정 |
|---|---|---|
| PatchTST | line 후보로 보류, direct band 후보는 탈락/보류 | 기존 q preset은 clean feature 기준 생존 실패 |
| TiDE | band 후보 1번 | q25 smoke에서는 coverage 부족 |
| CNN-LSTM | band 후보 2번, 빠른 smoke/calibration 실험용 | q25 smoke에서는 coverage 부족 |
| DLinear/NLinear | 다음 baseline 후보 | 이번 CP에서는 구현하지 않음 |
| NHITS/N-BEATS | Phase 1.5 후보 | 이번 CP 범위 밖 |

## 4. 실행 조건
공통 조건은 `feature_version=v3_adjusted_ohlc`, `timeframe=1D`, `seq_len=252`, `horizon=5`, `raw_future_return`, `checkpoint_selection=coverage_gate`, `batch_size=256`, `--no-compile`, `--no-wandb`, `--save-run` 미사용이다. q preset은 q25-b2(`q_low=0.25`, `q_high=0.75`, `lambda_band=2.0`)를 사용했다.

CNN-LSTM은 기존 CUDA bf16 안정화 이력에 따라 `--fp32-modules lstm,heads`를 붙였다. 이는 구조 변경이 아니라 runtime 안정화 옵션이다.

## 5. Band Smoke 결과
| 모델 | run_id | selected_epoch | selected_reason | coverage_gate_failed | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | band_loss | cross_loss | epoch seconds | VRAM MB | 밴드 판정 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TiDE | tide-1D-8f986c24d6ec | 3 | coverage_gate_failed_fallback_val_total | true | 0.426697 | 0.338113 | 0.235190 | 0.040327 | 0.067177 | 0.001944 | 26.83 | 387.84 | 탈락 |
| CNN-LSTM | cnn_lstm-1D-3e2a66260b30 | 3 | coverage_gate_failed_fallback_val_total | true | 0.631953 | 0.136222 | 0.231825 | 0.053698 | 0.034861 | 0.005635 | 73.41 | 1405.98 | 탈락 |

PatchTST q25-b2의 clean feature 기준 `avg_band_width`는 0.263445였다. TiDE와 CNN-LSTM은 폭 자체는 훨씬 좁지만, coverage가 낮고 upper breach가 0.15를 넘어서 밴드 후보로 생존하지 못했다.

## 6. Line 지표
| 모델 | mae | smape | direction_accuracy | spearman_ic | top_k_long_spread | top_k_short_spread | long_short_spread | 라인 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TiDE | 0.087004 | 1.559822 | 0.521871 | -0.019393 | 0.001705 | 0.000850 | 0.000855 | 탈락 |
| CNN-LSTM | 0.047710 | 1.494067 | 0.511138 | -0.065226 | -0.002148 | 0.001137 | -0.003284 | 탈락 |

라인 후보 기준은 `spearman_ic > 0`, `long_short_spread > 0`이다. TiDE는 IC가 음수였고, CNN-LSTM은 IC와 spread가 모두 음수였다.

## 7. Test 확인
| 모델 | test coverage | test upper_breach_rate | test spearman_ic | test long_short_spread | test fee_adjusted_return |
|---|---:|---:|---:|---:|---:|
| TiDE | 0.229044 | 0.278770 | -0.003557 | 0.001150 | -0.586264 |
| CNN-LSTM | 0.620170 | 0.222231 | -0.005643 | 0.003336 | 2.167809 |

test에서도 TiDE는 coverage가 크게 무너졌고, CNN-LSTM은 coverage 0.620170과 upper breach 0.222231로 밴드 기준을 통과하지 못했다. CNN-LSTM의 fee 지표는 양수지만 이번 CP에서는 보조 지표이므로 밴드 탈락 판정을 뒤집지 않는다.

## 8. Val/Test Gap
| 모델 | coverage gap | IC gap | spread gap | fee return gap |
|---|---:|---:|---:|---:|
| TiDE | 0.197653 | -0.015836 | -0.000296 | -0.241183 |
| CNN-LSTM | 0.011783 | -0.059583 | -0.006621 | -3.107834 |

TiDE는 validation과 test coverage 격차도 컸다. CNN-LSTM은 coverage gap은 작지만 validation/test 모두 밴드 coverage 범위 밖이다.

## 9. 판정
q25-b2 기준에서는 TiDE와 CNN-LSTM 모두 밴드 후보로 생존하지 못했다. 다만 이 결과는 q25-b2 단일 calibration smoke이며, 모델 자체 최종 폐기는 아니다. 다음에 TiDE/CNN-LSTM을 계속 볼 경우 q20-b2처럼 더 넓은 밴드 preset이나 lambda 조정만 최소 범위로 확인해야 한다.

## 10. 결론
CP34의 핵심 산출물은 평가판 분리다. PatchTST는 line 후보로만 보류하고, direct band 후보로는 현재 탈락/보류한다. TiDE와 CNN-LSTM은 q25-b2 smoke에서 band 후보 기준을 통과하지 못했다. DLinear/NLinear는 다음 baseline 후보지만 이번 CP에서는 구현하지 않았다.
