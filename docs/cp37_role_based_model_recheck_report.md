CP37은 모델 하나를 고르는 CP가 아니라, line model과 band model을 분리해 후보를 선발하는 CP다.

# CP37-M 역할 기반 모델 재평가 보고서

## 1. 목표
CP36에서 분리한 `line_gate`와 `band_gate`를 사용해 모델 역할별 후보를 다시 평가했다. 이번 CP는 단일 모델 성능 비교가 아니라 예측선 후보와 밴드 후보를 따로 선발하는 작업이다.

## 2. 공통 조건
공통 조건은 `feature_version=v3_adjusted_ohlc`, `timeframe=1D`, `horizon=5`, `batch_size=256`, `--no-compile`, `--no-wandb`, `--save-run` 미사용, 50티커 3epoch다. full 473티커, 신규 모델, DLinear/NLinear, 시그널 모델화는 하지 않았다.

CNN-LSTM은 기존 CUDA 안정화 이력에 따라 `--fp32-modules lstm,heads`를 유지했다. 이는 구조 변경이 아니라 runtime 안정화 옵션이다.

## 3. 실행 매트릭스
| 실험 | 역할 | 모델 | seq_len | q_low | q_high | lambda_band | band_mode | checkpoint_selection | exit code |
|---|---|---|---:|---:|---:|---:|---|---|---:|
| A | line_model | PatchTST | 252 | 0.25 | 0.75 | 2.0 | direct | line_gate | 0 |
| B | band_model | TiDE | 252 | 0.10 | 0.90 | 2.0 | param | band_gate | 0 |
| C | band_model | TiDE | 252 | 0.10 | 0.90 | 2.0 | direct | band_gate | 0 |
| D | band_model | CNN-LSTM | 120 | 0.20 | 0.80 | 2.0 | direct | band_gate | 0 |
| E | band_model | CNN-LSTM | 60 | 0.20 | 0.80 | 2.0 | direct | band_gate | 0 |

## 4. Line 후보 결과
| 실험 | run_id | selected_epoch | selected_reason | line_gate_pass | spearman_ic | long_short_spread | mae | smape | direction_accuracy | 판정 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| A | patchtst-1D-80cc132456a6 | 2 | line_gate_eligible | true | 0.073986 | 0.005177 | 0.071732 | 1.596081 | 0.469428 | 생존 |

PatchTST는 line 후보 기준을 통과했다. coverage는 0.981634로 과보수지만, line 후보 판정에서는 참고 지표로만 본다.

## 5. Band 후보 결과
| 실험 | run_id | selected_epoch | selected_reason | band_gate_pass | coverage | lower_breach | upper_breach | avg_band_width | 판정 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| B | tide-1D-1f9b2f31e428 | 3 | band_gate_eligible | true | 0.823158 | 0.114758 | 0.062084 | 0.107054 | 보류 |
| C | tide-1D-157a9dbe0ef0 | 1 | band_gate_eligible | true | 0.776830 | 0.152964 | 0.070205 | 0.160960 | 보류 |
| D | cnn_lstm-1D-60c908f98944 | 3 | band_gate_failed_fallback_val_total | false | 0.720917 | 0.115837 | 0.163246 | 0.064094 | 탈락 |
| E | cnn_lstm-1D-9f1bbaa7dac6 | 2 | band_gate_eligible | true | 0.755517 | 0.102432 | 0.142050 | 0.068763 | 보류 |

B, C, E는 validation 기준 `band_gate`를 통과했다. D는 coverage가 0.75 미만이고 upper breach가 0.15를 넘어 탈락이다.

## 6. Test Gap 확인
| 실험 | val coverage | test coverage | coverage gap | val upper | test upper | test spearman_ic | test spread | test fee |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.981634 | 0.980075 | 0.001559 | 0.013528 | 0.015325 | 0.012903 | 0.001692 | 0.121916 |
| B | 0.823158 | 0.589787 | 0.233371 | 0.062084 | 0.135763 | 0.005904 | 0.001658 | -0.500213 |
| C | 0.776830 | 0.529882 | 0.246948 | 0.070205 | 0.157517 | -0.012836 | -0.002357 | -0.892745 |
| D | 0.720917 | 0.662821 | 0.058096 | 0.163246 | 0.191043 | -0.036623 | -0.001908 | -0.606206 |
| E | 0.755517 | 0.698842 | 0.056675 | 0.142050 | 0.157765 | -0.024127 | -0.000159 | -0.234835 |

밴드 후보 B/C/E는 validation 기준으로는 통과했지만 test coverage가 낮다. 특히 B와 C는 coverage gap이 0.23 이상이라 즉시 full run 후보로 올리면 안 된다. E는 gap이 작지만 test coverage가 0.70 미만이고 upper breach가 0.15를 살짝 넘는다.

## 7. 속도와 VRAM
| 실험 | 평균 epoch seconds | VRAM peak MB |
|---|---:|---:|
| A | 77.95 | 5153.26 |
| B | 13.60 | 70.22 |
| C | 13.60 | 70.20 |
| D | 39.74 | 631.55 |
| E | 26.62 | 323.91 |

TiDE는 매우 가볍다. PatchTST는 가장 무겁지만 line 후보로는 지표가 가장 선명하다. CNN-LSTM seq60은 D보다 빠르고 validation band gate를 통과했지만 test 기준이 약하다.

## 8. stderr / 경고
B와 C에서 `ConstantInputWarning`이 stderr에 남았다. 종료코드는 0이며 학습은 완료됐다. 해당 경고는 일부 Spearman 계산 구간이 상수 입력이었다는 의미로, band 후보 판단에서는 탈락 조건으로 쓰지 않았다.

## 9. 판정
line 후보는 PatchTST를 생존으로 둔다. `line_gate_pass=true`, `spearman_ic=0.073986`, `long_short_spread=0.005177`이다.

band 후보는 B(TiDE param), C(TiDE direct), E(CNN-LSTM seq60)를 validation 기준 생존/보류로 둔다. 다만 test coverage가 낮아 full run 후보는 아니다. 역할 분리의 목적상 IC/spread 음수는 band 후보 탈락 근거로 쓰지 않았다.

## 10. 결론
CP37에서 역할 분리 판단은 작동했다. PatchTST는 line model 후보로 살아났고, TiDE/CNN-LSTM은 band model 후보로 일부 살아났지만 test 안정성이 부족해 보류다. 다음 단계는 full 473티커가 아니라 band 후보의 validation/test gap을 줄이는 split 안정성 또는 짧은 100티커 검증이다.
