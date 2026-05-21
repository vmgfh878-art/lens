# CP33-M PatchTST RevIN Ablation 보고서

## 1. 목표
PatchTST 기존 preset을 더 넓게 sweep하지 않고, 남은 핵심 설계 의심점인 RevIN output denormalize 영향을 50티커 3epoch 조건에서 분리했다. 동시에 다음 단계에서 DLinear/NLinear baseline 준비로 넘어갈지 판단했다.

## 2. 코드 변경
`ai/train.py`에 `--use-revin / --no-use-revin` CLI를 추가하고, `TrainConfig`와 PatchTST 생성 경로에 연결했다. `ai/inference.py`도 checkpoint config의 `use_revin` 값을 읽어 모델을 복원하도록 맞췄다. 새 모델 구조는 추가하지 않았다.

## 3. 공통 조건
공통 조건은 `model=patchtst`, `timeframe=1D`, `seq_len=252`, `patch_len=16`, `patch_stride=8`, `ci_aggregate=target`, `line_target_type=raw_future_return`, `band_target_type=raw_future_return`, `checkpoint_selection=coverage_gate`, `batch_size=256`, `--no-compile`, `--no-wandb`다. `--save-run`은 사용하지 않았다.

## 4. Feature Version 확인
네 run 모두 checkpoint config에서 `feature_version=v3_adjusted_ohlc`로 확인됐다. 입력 50티커 중 재무 gate로 `AMP`, `APA` 2개가 제외되어 실제 학습 대상은 48티커였다.

## 5. 실행 결과
| 실험 | 설정 | use_revin | selected_epoch | selected_reason | coverage_gate_failed | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | spearman_ic | long_short_spread | fee_adjusted_return | test fee | epoch seconds | exit code | 판정 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | q25-b2 | true | 3 | coverage_gate_eligible | false | 0.949298 | 0.015118 | 0.035584 | 0.263445 | 0.069912 | 0.005403 | 7.773737 | -0.103182 | 77.88 | 0 | 탈락 |
| B | q25-b2 | false | 3 | coverage_gate_failed_fallback_val_total | true | 0.992308 | 0.003005 | 0.004687 | 0.288923 | 0.013581 | -0.002669 | -0.955316 | -0.484688 | 77.44 | 0 | 탈락 |
| C | q30-b2 | false | 3 | coverage_gate_failed_fallback_val_total | true | 0.980636 | 0.008319 | 0.011045 | 0.229061 | 0.013253 | -0.002633 | -0.951315 | -0.376120 | 77.75 | 0 | 탈락 |
| D | q35-b2 | false | 3 | coverage_gate_failed_fallback_val_total | true | 0.962583 | 0.016232 | 0.021186 | 0.195214 | 0.016217 | -0.002276 | -0.938200 | -0.454023 | 77.60 | 0 | 탈락 |

## 6. Test 지표
| 실험 | coverage | upper_breach_rate | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|
| A q25-b2 RevIN on | 0.943217 | 0.042145 | 0.013758 | 0.001067 | -0.103182 |
| B q25-b2 RevIN off | 0.987830 | 0.006079 | -0.006993 | -0.000211 | -0.484688 |
| C q30-b2 RevIN off | 0.972190 | 0.013858 | -0.007819 | 0.000208 | -0.376120 |
| D q35-b2 RevIN off | 0.949738 | 0.025620 | -0.007623 | -0.000095 | -0.454023 |

## 7. Val/Test Gap
| 실험 | coverage gap | IC gap | spread gap | fee return gap |
|---|---:|---:|---:|---:|
| A q25-b2 RevIN on | 0.006082 | 0.056154 | 0.004335 | 7.876919 |
| B q25-b2 RevIN off | 0.004477 | 0.020574 | -0.002458 | -0.470628 |
| C q30-b2 RevIN off | 0.008445 | 0.021072 | -0.002841 | -0.575195 |
| D q35-b2 RevIN off | 0.012845 | 0.023840 | -0.002181 | -0.484177 |

## 8. 해석
RevIN off는 output denormalize로 인한 극단적 스케일 문제를 줄이는 방향이 아니라, 오히려 coverage를 0.96~0.99에 머물게 했다. q30/q35로 밴드를 좁혀도 coverage가 목표 구간으로 내려오지 않았고, validation long-short spread가 음수라 coverage gate를 통과하지 못했다.

## 9. 판정
판정 기준 1번은 충족하지 못했다. `use_revin=False`에서 coverage가 0.85~0.93으로 내려오지 않았고, IC/spread도 유지되지 않았다. 기준 2번에 따라 PatchTST 기존 preset 실험은 중단하는 것이 맞다. 생존 후보가 없으므로 CP34에서 DLinear/NLinear baseline 전환 준비가 필요하다.

## 10. 산출물
- 메트릭 JSON: `docs/cp33_patchtst_revin_ablation_metrics.json`
- 실행 로그: `docs/cp33_A_q25_b2_revin_on.stdout.log`
- 실행 로그: `docs/cp33_B_q25_b2_revin_off.stdout.log`
- 실행 로그: `docs/cp33_C_q30_b2_revin_off.stdout.log`
- 실행 로그: `docs/cp33_D_q35_b2_revin_off.stdout.log`
