CP53는 새 최고 성능을 찾는 CP가 아니라, CP52 표준 지표판으로 기존 후보 중 무엇을 믿고 다음 실험으로 갈지 정리하는 CP다.

## Executive Summary

기존 checkpoint/log 산출물을 재사용해 재학습 없이 forward-only 재채점을 수행했다. save-run, DB schema 변경, UI 수정, 모델 구조 변경, fake data 생성은 하지 않았다.

재채점 기준은 CP52 계약이다. line은 `ic_mean`, `ic_ir`, `spread_ir`, `false_safe_*`, `severe_downside_recall`로 해석했고, band는 단순 coverage가 아니라 `nominal_coverage` 대비 `coverage_abs_error`, asymmetric interval score, `band_width_ic`, `downside_width_ic`를 기준으로 해석했다. composite는 모델 탈락 기준이 아니라 제품 표시/정책 지표로 분리했다.

결론은 세 가지다. 첫째, h5 line 후보는 `h5_longer_context_seq252_p32_s16`이 가장 믿을 만하다. 둘째, band 후보는 CNN-LSTM `s60_q15_b2_direct` 계열만 생존권이고 TiDE param/direct는 dynamic downside width가 약해 보류다. 셋째, composite 정책은 `risk_first_upper_buffer_1.10`이 제품 표시 지표를 가장 안정화하지만, width 증가가 큰 정책 지표이므로 모델 성능 개선으로 해석하면 안 된다.

## Line 후보 재채점 표

아래 표는 test split 기준이다. h5/h10/h20은 같은 제품 후보로 직접 경쟁시키지 않고 branch별로 해석한다.

| 후보 | branch | 상태 | ic_mean | ic_ir | spread | spread/fee 해석 | false_safe_tail | false_safe_severe | severe_recall | downside_capture |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| h5_baseline_seq252_p16_s8 | h5 | 생존 | 0.0136 | 0.0559 | 0.0022 | 약한 양수 | 0.2214 | 0.2254 | 0.7746 | 0.2330 |
| h5_longer_context_seq252_p32_s16 | h5 | 후보 | 0.0241 | 0.1082 | 0.0034 | spread 후보권 | 0.1475 | 0.1486 | 0.8514 | 0.2439 |
| h5_dense_overlap_seq252_p16_s4 | h5 | 보류 | -0.0049 | -0.0186 | -0.0018 | line 약함 | 0.0079 | 0.0090 | 0.9910 | 0.2549 |
| h10_baseline_seq252_p16_s8 | h10 | 생존 | 0.0088 | 0.0341 | 0.0010 | 약한 양수 | 0.1880 | 0.1873 | 0.8127 | 0.2456 |
| h10_longer_context_seq252_p32_s16 | h10 | 탈락 | -0.0074 | -0.0289 | -0.0038 | 음수 | 0.2725 | 0.2759 | 0.7241 | 0.2419 |
| h10_dense_overlap_seq252_p16_s4 | h10 | 보류 | 0.0138 | 0.0585 | -0.0003 | spread 음수 | 0.0839 | 0.0828 | 0.9172 | 0.2655 |
| h20_baseline_seq252_p16_s8 | h20 | 탈락 | -0.0068 | -0.0338 | 0.0022 | IC 음수 | 0.2676 | 0.2679 | 0.7321 | 0.2295 |
| h20_longer_context_seq252_p32_s16 | h20 | 생존 | 0.0199 | 0.0820 | 0.0072 | spread 강함 | 0.3149 | 0.3132 | 0.6868 | 0.2366 |
| h20_dense_overlap_seq252_p16_s4 | h20 | 후보 | 0.0138 | 0.0653 | 0.0018 | 약한 양수 | 0.1091 | 0.1101 | 0.8899 | 0.2296 |
| h20_baseline_seq504_p16_s8 | h20 | 보류 | -0.0521 | -0.2176 | -0.0264 | 음수 | 0.0388 | 0.0396 | 0.9604 | 0.2403 |

## Band 후보 재채점 표

아래 표는 scalar calibration 적용 후보는 적용 후 test 기준이다. `coverage_abs_error`는 `empirical_coverage - nominal_coverage`의 절대값이다.

| 후보 | 평가 블록 | 상태 | nominal | empirical | coverage_abs_error | lower breach | upper breach | avg width | interval_score | band_width_ic | downside_width_ic |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s60_q20_b2_direct | scalar_width | 보류 | 0.60 | 0.7947 | 0.1947 | 0.1163 | 0.0890 | 0.1043 | 0.1422 | 0.2839 | 0.0479 |
| s60_q15_b2_direct | scalar_width | 생존 | 0.70 | 0.7978 | 0.0978 | 0.1187 | 0.0835 | 0.1073 | 0.1596 | 0.2502 | 0.0321 |
| s60_q15_b2_direct_188 | scalar_width | 생존 | 0.70 | 0.8069 | 0.1069 | 0.0844 | 0.1087 | 0.1021 | 0.1487 | 0.2463 | 0.0333 |
| tide_param_scalar_width | scalar_width | 보류 | 0.80 | 0.6315 | 0.1685 | 0.2148 | 0.1538 | 0.1149 | 0.8268 | 0.1809 | -0.0051 |
| tide_direct_original | original | 보류 | 0.80 | 0.5140 | 0.2860 | 0.3119 | 0.1742 | 0.1312 | 1.0960 | 0.0758 | -0.0025 |

## Composite 정책 재채점 표

Composite 지표는 모델 탈락 기준이 아니라 제품 표시와 정책 안정성 지표다. test split 기준이며, `composite_width_increase_ratio`는 raw composite 대비 폭 증가율이다.

| 정책 | line_inside | warning_rate | conservative_false_safe | coverage | lower breach | upper breach | width increase | interval_score | 해석 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_composite | 0.0000 | 0.7298 | 0.0000 | 0.7649 | 0.0191 | 0.2160 | 1.0000 | 0.2384 | 비교 기준. line 표시 정합성 실패 |
| risk_first_lower_preserve | 1.0000 | 0.0000 | 0.0000 | 0.8064 | 0.0021 | 0.1915 | 1.8270 | 0.3739 | 하방 보수성은 유지하지만 upper breach 높음 |
| risk_first_upper_buffer_1.10 | 1.0000 | 0.0000 | 0.0000 | 0.8957 | 0.0021 | 0.1021 | 1.9848 | 0.3834 | 표시 안정성은 가장 좋지만 폭 증가 큼 |

## h5/h10/h20 Branch 해석

h5는 현재 제품 line 후보 branch다. `h5_longer_context_seq252_p32_s16`은 IC와 spread가 양수이고 false-safe 계열도 가장 균형적이므로 다음 line 검증의 기본 후보로 유지한다. `h5_dense_overlap_seq252_p16_s4`는 IC/spread가 음수지만 false-safe가 매우 낮고 severe recall이 높아 risk-only 보조지표 후보로만 보류한다.

h10은 제품 기본 후보로 바로 올리기에는 약하다. baseline은 살아 있지만 edge가 작고, dense overlap은 risk 지표는 좋지만 spread가 음수다. h10은 h5 이후 별도 branch로만 유지한다.

h20은 제품 horizon 후보가 아니라 별도 branch다. h20 longer_context와 dense_overlap은 일부 지표가 살아 있지만 false-safe 또는 일반 line 지표 중 하나가 깨진다. h20 seq504는 일반 line 지표가 크게 음수라 탈락이다.

## 기존 판정에서 바뀐 후보 목록

`h5_longer_context_seq252_p32_s16`은 기존 판단과 동일하게 주력 h5 line 후보로 유지한다. CP52 기준에서도 spread와 false-safe 균형이 가장 낫다.

`h5_dense_overlap_seq252_p16_s4`는 일반 line 후보가 아니라 risk-only 보조 후보로 명확히 격하한다. IC/spread가 음수이므로 line model 후보로 부르면 안 된다.

CNN-LSTM `s60_q15_b2_direct_188`은 기존 CP45에서 생존 후보였지만, CP52 기준으로는 “후보”가 아니라 “생존”이다. nominal q15/q85의 0.70 coverage에 비해 empirical 0.8069라 보수적으로 넓고, downside_width_ic가 0.0333으로 약하다.

TiDE param/direct는 폐기가 아니라 보류다. coverage calibration과 downside dynamic width가 약하므로, 지금 composite band 주력으로 올리지는 않는다.

`risk_first_upper_buffer_1.10`은 composite 표시 정책 후보로 유지한다. 다만 width 증가율이 약 1.98배라 모델 성능 개선이 아니라 제품 표시 정책으로만 해석한다.

## 탈락/보류/생존 후보

Line 생존 또는 후보:

| 역할 | 후보 |
|---|---|
| h5 기본 line | h5_longer_context_seq252_p32_s16 |
| h5 기준선 | h5_baseline_seq252_p16_s8 |
| h5 risk-only 보조 | h5_dense_overlap_seq252_p16_s4 |
| h10 보류 | h10_baseline_seq252_p16_s8, h10_dense_overlap_seq252_p16_s4 |
| h20 branch 보류 | h20_longer_context_seq252_p32_s16, h20_dense_overlap_seq252_p16_s4 |

Band 생존 또는 보류:

| 상태 | 후보 |
|---|---|
| 생존 | s60_q15_b2_direct, s60_q15_b2_direct_188 |
| 보류 | s60_q20_b2_direct, tide_param_scalar_width, tide_direct_original |

탈락:

| 역할 | 후보 | 이유 |
|---|---|---|
| line | h10_longer_context_seq252_p32_s16 | IC/spread 음수, false-safe 높음 |
| line | h20_baseline_seq252_p16_s8 | IC 음수 |
| line | h20_baseline_seq504_p16_s8 | IC/spread 크게 음수 |

## 다음 CP 추천

1. 먼저 baseline band 비교로 간다. CP52에서 baseline 기준을 고정했으므로 Bollinger return band, historical quantile band, constant-width band를 같은 지표판에 올려야 CNN-LSTM band가 진짜 의미 있는지 판단할 수 있다.
2. band sweep을 계속한다면 `s60_q15_b2_direct_188`을 기준으로 downside_width_ic를 올리는 방향이어야 한다. 단순 coverage 개선이나 width 확장은 더 이상 목표가 아니다.
3. h20은 별도 branch로 유지한다. 제품 기본 horizon 후보와 섞지 말고, h20 전용 selector 또는 horizon bucket loss를 설계하기 전까지는 Phase 1.5 보류가 맞다.

## 검증 및 제한

재채점은 기존 checkpoint forward-only로 수행했다. 최초 실행은 결과 JSON 생성 후 CUDA teardown 단계에서 타임아웃으로 끊겼지만, `docs/cp53_existing_candidate_regrade_metrics.json`은 JSON 무결성 검사를 통과했다. 이후 CP53 스크립트에 명시적 CUDA cleanup을 추가했다.

CP52의 일부 지표는 기존 aggregate 로그만으로는 계산할 수 없어 checkpoint를 다시 forward 평가했다. 새 학습, save-run, DB 저장, UI 수정, 모델 구조 변경은 수행하지 않았다.
