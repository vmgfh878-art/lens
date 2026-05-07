CP35는 TiDE/CNN-LSTM 폐기가 아니라, q25-b2 실패 이후 모델별 구조에 맞춘 최소 rescue smoke다.

# CP35-M TiDE/CNN-LSTM Band Rescue Smoke 보고서

## 1. 목표
CP34에서 q25-b2 단일 조건으로 실패한 TiDE와 CNN-LSTM을 즉시 폐기하지 않고, 모델별 특성에 맞춘 최소 rescue smoke를 수행했다. 이번 CP는 대형 튜닝이 아니라 살릴 여지가 있는지 확인하는 작업이다.

## 2. 공통 조건
공통 조건은 `feature_version=v3_adjusted_ohlc`, `timeframe=1D`, `horizon=5`, `raw_future_return`, `checkpoint_selection=coverage_gate`, `batch_size=256`, `--no-compile`, `--no-wandb`, `--save-run` 미사용, 50티커 3epoch다. fee 지표는 보조로만 기록했다.

CNN-LSTM은 기존 CUDA 안정화 이력에 따라 `--fp32-modules lstm,heads`를 유지했다. 이는 구조 변경이 아니라 runtime 안정화 옵션이다.

## 3. 실행 매트릭스
| 실험 | 모델 | seq_len | q_low | q_high | lambda_band | band_mode | 목적 | exit code |
|---|---|---:|---:|---:|---:|---|---|---:|
| A | TiDE | 252 | 0.15 | 0.85 | 2.0 | direct | q25가 너무 공격적이었는지 확인 | 0 |
| B | TiDE | 252 | 0.10 | 0.90 | 2.0 | direct | TiDE가 coverage를 회복할 수 있는지 확인 | 0 |
| C | CNN-LSTM | 120 | 0.20 | 0.80 | 2.0 | direct | 짧은 local pattern 안정성 확인 | 0 |
| D | CNN-LSTM | 60 | 0.20 | 0.80 | 2.0 | direct | local volatility band 후보성 확인 | 0 |
| E | TiDE | 252 | 0.10 | 0.90 | 2.0 | param | positive-width 계열 1회 추가 확인 | 0 |

현재 CLI에는 `positive_width`라는 `band_mode`가 없고 `direct`, `param`만 존재한다. `param`은 `center ± exp(log_half_width)` 구조라 양수 폭을 강제하므로 E 실험의 대체 구현으로 사용했다.

## 4. Validation 결과
| 실험 | run_id | selected_epoch | selected_reason | coverage_gate_failed | coverage | lower_breach | upper_breach | avg_band_width | band_loss | cross_loss |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A | tide-1D-ba602ce49791 | 3 | coverage_gate_failed_fallback_val_total | true | 0.618807 | 0.229075 | 0.152117 | 0.065644 | 0.019441 | 0.000027 |
| B | tide-1D-bccd7400132a | 3 | coverage_gate_failed_fallback_val_total | true | 0.726848 | 0.170832 | 0.102320 | 0.087572 | 0.015985 | 0.000042 |
| C | cnn_lstm-1D-e86b7764b762 | 3 | coverage_gate_failed_fallback_val_total | true | 0.720917 | 0.115837 | 0.163246 | 0.064094 | 0.019196 | 0.000124 |
| D | cnn_lstm-1D-d22d8df596cc | 3 | coverage_gate_failed_fallback_val_total | true | 0.727251 | 0.102873 | 0.169876 | 0.062560 | 0.018481 | 0.000206 |
| E | tide-1D-53f8fe3ea4b1 | 3 | coverage_gate_failed_fallback_val_total | true | 0.823158 | 0.114758 | 0.062084 | 0.107054 | 0.015372 | 0.000000 |

## 5. Line/투자 지표
| 실험 | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | 판정 |
|---|---:|---:|---:|---:|---|
| A | -0.027516 | -0.000579 | -0.928985 | 0.531152 | line 탈락 |
| B | -0.042241 | -0.002431 | -0.978617 | 0.519283 | line 탈락 |
| C | -0.039287 | -0.000196 | -0.621025 | 0.467176 | line 탈락 |
| D | -0.043366 | -0.000577 | -0.738586 | 0.471459 | line 탈락 |
| E | -0.027770 | -0.001251 | -0.958327 | 0.512122 | line 탈락 |

모든 실험에서 validation 기준 `spearman_ic > 0`, `long_short_spread > 0`을 만족하지 못했다.

## 6. Test 확인
| 실험 | test coverage | test upper_breach | test spearman_ic | test long_short_spread | test fee_adjusted_return |
|---|---:|---:|---:|---:|---:|
| A | 0.346279 | 0.238873 | -0.005301 | 0.000105 | -0.722939 |
| B | 0.435437 | 0.207639 | -0.005646 | 0.000612 | -0.665393 |
| C | 0.662821 | 0.191043 | -0.036623 | -0.001908 | -0.606206 |
| D | 0.676216 | 0.178797 | -0.020096 | 0.000350 | -0.043332 |
| E | 0.589787 | 0.135763 | 0.005904 | 0.001658 | -0.500213 |

E는 validation band 기준으로는 가장 살아났지만, test coverage가 0.589787로 낮다. 따라서 full run 후보로 올리기에는 아직 부족하다.

## 7. 속도와 VRAM
| 실험 | 평균 epoch seconds | VRAM peak MB |
|---|---:|---:|
| A | 14.86 | 70.20 |
| B | 13.48 | 70.20 |
| C | 40.40 | 631.55 |
| D | 26.15 | 323.91 |
| E | 13.70 | 70.22 |

TiDE는 매우 가볍게 돌았고, CNN-LSTM은 seq_len 60에서 속도와 VRAM이 개선됐다. 다만 성능 기준은 둘 다 통과하지 못했다.

## 8. stderr / 경고
B와 E에서 `ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.` 경고가 stderr에 남았다. 종료코드는 0이며 학습 자체는 완료됐다. 해당 경고는 일부 Spearman 계산 구간의 입력이 상수였다는 뜻이라, line/rank 신호가 약하다는 해석과도 일치한다.

## 9. 판정
밴드 기준으로 A~D는 탈락이다. E는 validation 기준 `coverage=0.823158`, `upper_breach=0.062084`, `lower_breach=0.114758`로 band rescue 가능성을 보였지만, `coverage_gate_failed=true`이며 line 지표가 실패했다. 또한 test coverage가 낮아 강한 생존 후보가 아니라 약한 보류 후보다.

라인 기준으로는 A~E 모두 탈락이다.

## 10. 결론
CP35 결과만으로 TiDE/CNN-LSTM을 full run에 올리면 안 된다. 다만 TiDE `q10-b2 + param`은 band 전용 관점에서는 약한 보류 후보로 남길 수 있다. 현 `coverage_gate`가 line 지표까지 포함해 band 후보를 fallback 처리하므로, 다음 단계에서 계속 밴드 후보를 보려면 밴드 전용 checkpoint gate 분리가 먼저 필요하다.
