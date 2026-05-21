# CP27-R Coverage-Aware Checkpoint Selection 보고서

## 1. 목표
`val_total`만 낮은 epoch를 저장하면서 밴드 coverage가 무너지는 문제를 수정했다. PatchTST의 밴드 목적에 맞게 `coverage_gate` checkpoint selector를 추가하고, q25/q20/q15 100티커 조건을 재평가했다.

## 2. 구현 변경
- `ai/train.py`: `--checkpoint-selection {val_total,coverage_gate}` CLI를 추가했다. 기본값은 기존 동작 보존을 위해 `val_total`이다.
- `ai/train.py`: epoch별 validation metric으로 legacy best와 coverage-gate best 후보만 보관하고, 학습 종료 시 selector가 고른 checkpoint를 복원한다. 모든 epoch의 `state_dict`를 쌓지 않도록 메모리 사용을 제한했다.
- `ai/train.py`: `evaluate_bundle` 반환값에 `lower_breach_rate`, `upper_breach_rate`를 포함하도록 보강했다.
- `ai/tests/test_checkpoint_selection.py`: coverage gate eligible, fallback, legacy selector 테스트를 추가했다.

## 3. Coverage Gate 정책
eligible 조건은 `coverage ∈ [0.75, 0.95]`, `upper_breach_rate <= 0.15`, `lower_breach_rate <= 0.20`, `spearman_ic > 0`, `long_short_spread > 0`이다. eligible checkpoint가 있으면 `upper_breach_rate`가 낮은 후보를 우선하고, 그 다음 `spearman_ic`, `long_short_spread`, `val_total` 순서로 정렬한다.

## 4. 실행 조건
공통 조건은 `patchtst`, `1D`, `seq_len=252`, `batch_size=256`, `epochs=3`, `limit_tickers=100`, `patch_len=16`, `patch_stride=8`, `band_mode=direct`, `lambda_band=2.0`, `checkpoint_selection=coverage_gate`이다. q15-b2는 별도 `epochs=2` smoke도 실행했다.

## 5. 선택 Epoch 비교
| 후보 | selected_epoch | best_val_total_epoch | selected_reason | coverage_gate_failed | val_total(best 선택 전) | selected coverage | selected upper | selected lower | selected fee return |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| q25-b2 | 1 | 3 | coverage_gate_eligible | false | 0.121986 | 0.889698 | 0.081341 | 0.028961 | 0.479462 |
| q20-b2 | 1 | 3 | coverage_gate_eligible | false | 0.107455 | 0.924373 | 0.053875 | 0.021752 | 0.793004 |
| q15-b2 | 2 | 3 | coverage_gate_eligible | false | 0.056952 | 0.802552 | 0.095257 | 0.102191 | 0.545685 |
| q15-b2 epochs=2 | 2 | 2 | coverage_gate_eligible | false | 0.065535 | 0.886805 | 0.059548 | 0.053648 | -0.214170 |

## 6. Validation 지표
| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.889698 | 0.028961 | 0.081341 | 0.222528 | 0.059091 | 0.024434 | 0.002503 | 0.479462 | 0.480827 | 0.281766 | 0.041085 |
| q20-b2 | 0.924373 | 0.021752 | 0.053875 | 0.248139 | 0.051846 | 0.024377 | 0.002717 | 0.793004 | 0.482232 | 0.273959 | 0.040565 |
| q15-b2 | 0.802552 | 0.102191 | 0.095257 | 0.144704 | 0.027121 | 0.018834 | 0.002331 | 0.545685 | 0.481724 | 0.397908 | 0.041776 |
| q15-b2 epochs=2 | 0.886805 | 0.053648 | 0.059548 | 0.188191 | 0.031262 | 0.012599 | 0.001433 | -0.214170 | 0.473445 | 0.334296 | 0.040465 |

## 7. Test 지표
| 후보 | coverage | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.875978 | 0.236314 | 0.027993 | 0.035308 | 0.004973 | 3.472470 | 0.503154 | 0.289091 | 0.042581 |
| q20-b2 | 0.912518 | 0.263650 | 0.029175 | 0.035644 | 0.005515 | 4.546744 | 0.504948 | 0.279022 | 0.042309 |
| q15-b2 | 0.777554 | 0.153990 | 0.022322 | 0.043897 | 0.006738 | 8.187150 | 0.506592 | 0.406984 | 0.043548 |
| q15-b2 epochs=2 | 0.868803 | 0.200061 | 0.023954 | 0.038981 | 0.004444 | 2.695262 | 0.503214 | 0.342426 | 0.042011 |

## 8. CP26 대비 변화
| 후보 | CP26 val_total 선택 coverage | CP26 fee return | CP27 선택 coverage | CP27 fee return | 판단 |
|---|---:|---:|---:|---:|---|
| q25-b2 | 0.544565 | -0.206959 | 0.889698 | 0.479462 | 기본형 후보로 복구 |
| q20-b2 | 0.650025 | -0.118567 | 0.924373 | 0.793004 | 보수 기본형 후보로 복구 |
| q15-b2 | 0.749335 | 0.314993 | 0.802552 | 0.545685 | 공격형 후보로 개선 |

## 9. 판정
q20-b2와 q25-b2는 둘 다 기본형 후보 조건을 통과했다. q20-b2는 coverage와 upper breach가 더 안정적이고 test fee return도 높아 보수 기본형에 가깝다. q25-b2는 밴드 폭이 더 좁고 coverage도 목표 구간 안이라 실사용 기본형 후보로 유지할 수 있다.

## 10. q15-b2 해석
q15-b2 3epoch run에서는 coverage gate가 epoch2를 선택했고, validation/test 모두 공격형 후보 조건을 통과했다. 별도 2epoch smoke는 max epoch 변경으로 LR schedule이 달라져 validation fee return이 음수였으므로, "2epoch 고정"보다 "3epoch 실행 + coverage_gate epoch2 선택"을 채택하는 쪽이 맞다.

## 11. 검증
- `python -m unittest ai.tests.test_checkpoint_selection ai.tests.test_evaluation_targets ai.tests.test_patchtst_cli_config`: 12 tests OK.
- q25-b2, q20-b2, q15-b2, q15-b2 epochs=2 CUDA run 모두 exit code 0.
- stderr 로그 4건 모두 비어 있음.

## 12. 산출물
- 메트릭 JSON: `docs/cp27_coverage_aware_checkpoint_metrics.json`
- 로그: `docs/cp27_A_q25_b2_coverage_gate_100_1d.stdout.log`
- 로그: `docs/cp27_B_q20_b2_coverage_gate_100_1d.stdout.log`
- 로그: `docs/cp27_C_q15_b2_coverage_gate_100_1d.stdout.log`
- 로그: `docs/cp27_D_q15_b2_coverage_gate_100_1d_epochs2.stdout.log`

## 13. 결론
CP27의 가설은 맞았다. PatchTST가 깨진 것이 아니라, 좋은 coverage epoch를 버리고 val_total이 낮은 과소 coverage epoch를 저장하는 checkpoint 선택 문제가 핵심 병목이었다. 다음 후보는 q20-b2를 보수 기본형, q25-b2를 기본형, q15-b2를 공격형으로 두고 더 큰 ticker 안정성 검증으로 넘어갈 수 있다.
