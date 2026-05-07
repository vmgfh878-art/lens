CP32는 모델 개선이 아니라 v3_adjusted_ohlc 이후 기존 PatchTST 판단을 재검증하는 CP다.

# CP32-M PatchTST clean feature 재검증 보고서

## 1. 목표
CP29-D, CP30-G, CP31-S 이후 `feature_version=v3_adjusted_ohlc`와 저장 계약이 정리된 상태에서 기존 PatchTST 후보를 재검증했다. 이전 CP20~28 결과는 v2 오염 피처 기준이므로 참고 로그로만 둔다.

## 2. 고정 조건
공통 조건은 `model=patchtst`, `timeframe=1D`, `seq_len=252`, `patch_len=16`, `patch_stride=8`, `ci_aggregate=target`, `line_target_type=raw_future_return`, `band_target_type=raw_future_return`, `checkpoint_selection=coverage_gate`, `batch_size=256`, `--no-compile`, `--no-wandb`다. `--save-run`은 사용하지 않았다.

## 3. Feature Version 확인
네 run 모두 checkpoint config에서 `feature_version=v3_adjusted_ohlc`로 확인됐다. 따라서 이번 결과는 CP29-D 이후 clean feature 계약 기준으로 해석한다.

## 4. 50티커 실행 결과
입력 50티커 중 재무 gate로 `AMP`, `APA` 2개가 제외되어 실제 학습 대상은 48티커였다. 네 후보 모두 exit code 0으로 종료했다.

| 후보 | run_id | selected_epoch | selected_reason | coverage_gate_failed | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | spearman_ic | long_short_spread | fee_adjusted_return | test fee | epoch seconds | VRAM MB | 판정 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline q10-b1 | patchtst-1D-423ff0419420 | 3 | coverage_gate_failed_fallback_val_total | true | 0.999977 | 0.000012 | 0.000012 | 0.646555 | 0.071140 | 0.005718 | 9.368415 | -0.333283 | 77.48 | 5153.26 | 탈락 |
| q20-b2 | patchtst-1D-70d1130a61d5 | 3 | coverage_gate_failed_fallback_val_total | true | 0.984291 | 0.004490 | 0.011219 | 0.333313 | 0.067860 | 0.005278 | 6.990899 | -0.266205 | 78.11 | 5153.26 | 탈락 |
| q25-b2 | patchtst-1D-5c116c06a638 | 3 | coverage_gate_eligible | false | 0.949298 | 0.015118 | 0.035584 | 0.263445 | 0.069912 | 0.005403 | 7.773737 | -0.103182 | 78.00 | 5153.26 | 탈락 |
| q15-b2 | patchtst-1D-36f53d045620 | 3 | coverage_gate_failed_fallback_val_total | true | 0.995788 | 0.001473 | 0.002738 | 0.410676 | 0.068076 | 0.005378 | 7.482988 | -0.222129 | 77.91 | 5153.26 | 탈락 |

## 5. Val/Test 격차
validation에서는 모든 후보의 IC와 spread가 양수였지만, test fee가 전부 음수였다. 특히 validation fee와 test fee의 격차가 7~10포인트 수준으로 커서 50티커 결과만으로 다음 단계로 올릴 수 없다.

| 후보 | coverage gap | IC gap | spread gap | fee return gap |
|---|---:|---:|---:|---:|
| baseline q10-b1 | 0.000140 | 0.056450 | 0.005345 | 9.701698 |
| q20-b2 | 0.003459 | 0.054014 | 0.004703 | 7.257104 |
| q25-b2 | 0.006082 | 0.056154 | 0.004335 | 7.876919 |
| q15-b2 | 0.000796 | 0.054201 | 0.004642 | 7.705117 |

## 6. 후보별 해석
baseline, q20-b2, q15-b2는 coverage가 0.98~1.00에 가까워 coverage gate 실패 fallback으로 저장됐다. 즉 밴드가 너무 넓고 기본형/공격형 후보가 아니다.

q25-b2는 coverage gate 자체는 통과했지만 coverage 0.949298로 기본형 기준 상단 0.93을 넘고, test fee가 -0.103182로 음수였다. 그래서 100티커로 올리지 않았다.

## 7. 100티커 / 200티커 단계
50티커에서 살아남은 후보가 없으므로 100티커 안정성 단계는 실행하지 않았다. 따라서 200티커 제한 확인도 실행하지 않았다. 이는 지시서의 “50티커에서 살아남은 후보만 100티커 실행” 조건에 따른 중단이다.

## 8. 판정
clean feature 기준에서도 기존 PatchTST 후보는 통과하지 못했다. CP27에서 보였던 후보 복구는 v2 오염 피처와 당시 universe 조건에 의존했을 가능성이 크며, `v3_adjusted_ohlc` 기준에서는 밴드가 과도하게 보수적이고 test 수익 지표가 약하다.

## 9. 다음 제안
PatchTST 구조를 더 만지는 것보다, 먼저 같은 clean feature와 같은 저장 계약에서 단순 baseline을 붙이는 편이 낫다. 다만 CP32 지시상 DLinear/NLinear 비교는 금지였으므로 이번 CP에서는 착수하지 않았다.

## 10. 산출물
- 메트릭 JSON: `docs/cp32_patchtst_clean_feature_revalidation_metrics.json`
- 실행 로그: `docs/cp32_50_A_baseline_q10_b1.stdout.log`
- 실행 로그: `docs/cp32_50_B_q20_b2.stdout.log`
- 실행 로그: `docs/cp32_50_C_q25_b2.stdout.log`
- 실행 로그: `docs/cp32_50_D_q15_b2.stdout.log`
