# CP148-LM-1D Stage 4-0 risk-aware rerank 보고서

- 생성 시각: 2026-05-09T02:30:40
- 범위: 기존 CP148 Stage 2/3 line_model 후보 리랭크
- 금지 준수: 새 학습 없음, save-run 없음, DB write 없음, inference 저장 없음, product promotion 없음, live fetch 없음, band/composite 실험 없음
- 기준: beta=2.0 유지, 기존 line_gate는 생존 조건으로만 사용
- Stage 2 입력: `docs\cp_archive\model_line\cp148_lm_1d_stage0_2_metrics.json`
- Stage 3 입력: `docs\cp148_lm_1d_stage3_false_safe_sweep_metrics.json`
- stress/bucket 분해: validation split forward-only 재평가

## 1. 기준값

| 항목 | 값 |
| --- | --- |
| baseline IC SOTA | 0.057514 |
| baseline spread SOTA | 0.009231 |
| baseline false_safe SOTA | 0.458524 |
| baseline severe_recall SOTA | 0.536975 |
| existing product false_safe | 0.359558 |
| existing product severe_recall | 0.624853 |
| Stage 2 best false_safe | 0.308661 |
| Stage 2 best severe_recall | 0.685019 |

## 2. 기존 Stage 2 순위

| 기존순위 | 후보 | score | IC | spread | fee | false_safe | severe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cp148_s2_patchtst_no_fund_p32_s16 | 0.760000 | 0.056972 | 0.008389 | 22.429806 | 0.308661 | 0.685019 |
| 2 | cp148_s2_patchtst_pvv_p16_s8 | 0.650000 | 0.060282 | 0.010404 | 108.098406 | 0.319118 | 0.677274 |
| 3 | cp148_s2_patchtst_pvv_p32_s16 | 0.550000 | 0.051545 | 0.008207 | 20.433456 | 0.313708 | 0.679879 |
| 4 | cp148_s2_cnn_lstm_pvv | 0.480000 | 0.006532 | -0.000158 | -0.873278 | 0.113770 | 0.865938 |
| 5 | cp148_s2_patchtst_pvv_seq180 | 0.280000 | 0.038864 | 0.007717 | 20.496523 | 0.467943 | 0.522780 |
| 6 | cp148_s2_tide_pvv | 0.280000 | 0.034822 | 0.004852 | 1.355016 | 0.318908 | 0.672817 |

## 3. 새 risk-aware 순위

| 새순위 | 후보 | stage | 분류 | 라벨 | IC | spread | fee | false_safe | severe | stress FS | stress severe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cp148_s2_patchtst_no_fund_p32_s16 | stage2 | line_survivor | primary_stage4_base | 0.056972 | 0.008389 | 22.429806 | 0.308661 | 0.685019 | 0.206490 | 0.651515 |
| 2 | cp148_s2_patchtst_pvv_p32_s16 | stage2 | line_survivor | secondary_stage4_base | 0.051545 | 0.008207 | 20.433456 | 0.313708 | 0.679879 | 0.197640 | 0.681818 |
| 3 | cp148_s2_tide_pvv | stage2 | line_survivor | stress_risk_candidate | 0.034822 | 0.004852 | 1.355016 | 0.318908 | 0.672817 | 0.356932 | 0.803030 |
| 4 | cp148_s2_patchtst_pvv_p16_s8 | stage2 | line_survivor | stress_risk_candidate | 0.060282 | 0.010404 | 108.098406 | 0.319118 | 0.677274 | 0.277286 | 0.606061 |
| 5 | s3_no_fund_d15_wd01_lr1e4 | stage3 | line_survivor | stress_risk_candidate | 0.067490 | 0.010936 | 193.558523 | 0.329803 | 0.657569 | 0.256637 | 0.666667 |
| 6 | s3_no_fund_d25_wd10_lr2e4 | stage3 | line_survivor | stress_risk_candidate | 0.064434 | 0.009902 | 98.356080 | 0.335761 | 0.654493 | 0.259587 | 0.666667 |
| 7 | s3_no_fund_d25_wd05_lr1e4 | stage3 | line_survivor | stress_risk_candidate | 0.065752 | 0.010534 | 151.616428 | 0.358881 | 0.628424 | 0.286136 | 0.636364 |
| 8 | s3_pvv_p16_d35_wd10_lr5e5 | stage3 | line_survivor | stress_risk_candidate | 0.073080 | 0.016486 | 12216.557601 | 0.393621 | 0.590096 | 0.253687 | 0.651515 |
| 9 | s3_pvv_p16_d25_wd05_lr1e4 | stage3 | line_survivor | stress_risk_candidate | 0.072878 | 0.016272 | 10493.350737 | 0.402969 | 0.581938 | 0.265487 | 0.651515 |
| 10 | s3_pvv_p16_d15_wd01_lr1e4 | stage3 | line_survivor | stress_risk_candidate | 0.072843 | 0.016174 | 9622.676116 | 0.407521 | 0.568441 | 0.259587 | 0.651515 |
| 11 | s3_pvv_p32_ref_d25_wd05_lr1e4 | stage3 | line_survivor | stress_risk_candidate | 0.070157 | 0.012709 | 737.377406 | 0.418287 | 0.565963 | 0.277286 | 0.636364 |
| 12 | s3_no_fund_d35_wd10_lr5e5 | stage3 | line_survivor | stress_risk_candidate | 0.054787 | 0.012445 | 701.104653 | 0.432054 | 0.552708 | 0.297935 | 0.621212 |
| 13 | cp148_s2_patchtst_pvv_seq180 | stage2 | line_survivor | rejected | 0.038864 | 0.007717 | 20.496523 | 0.467943 | 0.522780 | 0.417763 | 0.533333 |

### 순위 변경 이유

- 기존 Stage 2 score는 alpha와 risk를 함께 본 composite였지만, Stage 4-0은 false_safe_tail_rate를 1순위로 둔다.
- CNN-LSTM은 false_safe/severe가 가장 좋지만 line_gate, spread, fee가 실패해 risk_only_reference로 분리했다.
- PatchTST no_fund p32/s16은 false_safe와 severe가 Stage 2 전체 최고라 primary_stage4_base로 유지한다.
- PatchTST pvv p32/s16은 pvv p16/s8보다 alpha는 약하지만 false_safe가 낮아 secondary_stage4_base로 올라갔다.
- pvv p16/s8과 Stage 3 후보들은 IC/spread/fee가 강하지만 전체 false_safe가 no_fund/pvv p32보다 약해 base가 아니라 stress_risk_candidate로 분리했다.

## 4. 전체 지표표

| 후보 | stage | line_gate | category | label | IC | spread | fee | FS | severe | bias | sacrifice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cp148_s2_cnn_lstm_pvv | stage2 | False | risk_only_reference | risk_only_reference | 0.006532 | -0.000158 | -0.873278 | 0.113770 | 0.865938 | -0.008075 | 0.049577 |
| cp148_s2_patchtst_no_fund_p32_s16 | stage2 | True | line_survivor | primary_stage4_base | 0.056972 | 0.008389 | 22.429806 | 0.308661 | 0.685019 | -0.021586 | 0.064156 |
| cp148_s2_patchtst_pvv_p16_s8 | stage2 | True | line_survivor | stress_risk_candidate | 0.060282 | 0.010404 | 108.098406 | 0.319118 | 0.677274 | -0.020512 | 0.063191 |
| cp148_s2_patchtst_pvv_p32_s16 | stage2 | True | line_survivor | secondary_stage4_base | 0.051545 | 0.008207 | 20.433456 | 0.313708 | 0.679879 | -0.021091 | 0.063878 |
| cp148_s2_patchtst_pvv_seq180 | stage2 | True | line_survivor | rejected | 0.038864 | 0.007717 | 20.496523 | 0.467943 | 0.522780 | -0.004028 | 0.044912 |
| cp148_s2_tide_pvv | stage2 | True | line_survivor | stress_risk_candidate | 0.034822 | 0.004852 | 1.355016 | 0.318908 | 0.672817 | -0.006280 | 0.047002 |
| s3_no_fund_d15_wd01_lr1e4 | stage3 | True | line_survivor | stress_risk_candidate | 0.067490 | 0.010936 | 193.558523 | 0.329803 | 0.657569 | -0.019523 | 0.061008 |
| s3_no_fund_d25_wd05_lr1e4 | stage3 | True | line_survivor | stress_risk_candidate | 0.065752 | 0.010534 | 151.616428 | 0.358881 | 0.628424 | -0.015785 | 0.057007 |
| s3_no_fund_d25_wd10_lr2e4 | stage3 | True | line_survivor | stress_risk_candidate | 0.064434 | 0.009902 | 98.356080 | 0.335761 | 0.654493 | -0.018642 | 0.060305 |
| s3_no_fund_d35_wd10_lr5e5 | stage3 | True | line_survivor | stress_risk_candidate | 0.054787 | 0.012445 | 701.104653 | 0.432054 | 0.552708 | -0.008219 | 0.048192 |
| s3_pvv_p16_d15_wd01_lr1e4 | stage3 | True | line_survivor | stress_risk_candidate | 0.072843 | 0.016174 | 9622.676116 | 0.407521 | 0.568441 | -0.012533 | 0.051543 |
| s3_pvv_p16_d25_wd05_lr1e4 | stage3 | True | line_survivor | stress_risk_candidate | 0.072878 | 0.016272 | 10493.350737 | 0.402969 | 0.581938 | -0.011668 | 0.052497 |
| s3_pvv_p16_d35_wd10_lr5e5 | stage3 | True | line_survivor | stress_risk_candidate | 0.073080 | 0.016486 | 12216.557601 | 0.393621 | 0.590096 | -0.012898 | 0.053708 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | stage3 | True | line_survivor | stress_risk_candidate | 0.070157 | 0.012709 | 737.377406 | 0.418287 | 0.565963 | -0.010181 | 0.050310 |

## 5. stress regime별 지표표

| 후보 | 구간 | samples | dates | FS | severe | spread | fee |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cp148_s2_patchtst_no_fund_p32_s16 | calm | 96330 | 249 | 0.312384 | 0.672358 | 0.001277 | -0.005913 |
| cp148_s2_patchtst_no_fund_p32_s16 | neutral | 73366 | 414 | 0.292287 | 0.688036 | 0.005514 | 2.231335 |
| cp148_s2_patchtst_no_fund_p32_s16 | stress | 1667 | 24 | 0.206490 | 0.651515 | 0.026954 | 0.805512 |
| cp148_s2_patchtst_no_fund_p32_s16 | vix_rising | 78929 | 316 | 0.309427 | 0.683759 | 0.008110 | 4.802388 |
| cp148_s2_patchtst_no_fund_p32_s16 | breadth_worsening | 87708 | 357 | 0.293270 | 0.700055 | 0.003094 | 0.406264 |
| cp148_s2_patchtst_pvv_p32_s16 | calm | 96330 | 249 | 0.314757 | 0.664524 | 0.001369 | 0.028773 |
| cp148_s2_patchtst_pvv_p32_s16 | neutral | 73366 | 414 | 0.302391 | 0.682500 | 0.007343 | 5.873216 |
| cp148_s2_patchtst_pvv_p32_s16 | stress | 1667 | 24 | 0.197640 | 0.681818 | 0.020026 | 0.537124 |
| cp148_s2_patchtst_pvv_p32_s16 | vix_rising | 78929 | 316 | 0.313132 | 0.683759 | 0.009324 | 7.680169 |
| cp148_s2_patchtst_pvv_p32_s16 | breadth_worsening | 87708 | 357 | 0.295587 | 0.694326 | 0.003291 | 0.466752 |
| cp148_s2_tide_pvv | calm | 96330 | 249 | 0.281889 | 0.723383 | 0.001239 | -0.258002 |
| cp148_s2_tide_pvv | neutral | 73366 | 414 | 0.326642 | 0.683750 | 0.004395 | 0.666830 |
| cp148_s2_tide_pvv | stress | 1667 | 24 | 0.356932 | 0.803030 | 0.002500 | -0.027267 |
| cp148_s2_tide_pvv | vix_rising | 78929 | 316 | 0.302079 | 0.686809 | 0.004832 | 0.732730 |
| cp148_s2_tide_pvv | breadth_worsening | 87708 | 357 | 0.304232 | 0.736278 | 0.002352 | -0.171304 |
| cp148_s2_patchtst_pvv_p16_s8 | calm | 96330 | 249 | 0.333695 | 0.647047 | 0.001079 | -0.044106 |
| cp148_s2_patchtst_pvv_p16_s8 | neutral | 73366 | 414 | 0.287908 | 0.698214 | 0.008693 | 11.610590 |
| cp148_s2_patchtst_pvv_p16_s8 | stress | 1667 | 24 | 0.277286 | 0.606061 | 0.021936 | 0.582723 |
| cp148_s2_patchtst_pvv_p16_s8 | vix_rising | 78929 | 316 | 0.324939 | 0.666984 | 0.009967 | 9.787162 |
| cp148_s2_patchtst_pvv_p16_s8 | breadth_worsening | 87708 | 357 | 0.302255 | 0.695805 | 0.005504 | 2.265099 |
| s3_no_fund_d15_wd01_lr1e4 | calm | 96330 | 249 | 0.351496 | 0.619928 | 0.001671 | 0.184234 |
| s3_no_fund_d15_wd01_lr1e4 | neutral | 73366 | 414 | 0.286898 | 0.678214 | 0.008701 | 14.306422 |
| s3_no_fund_d15_wd01_lr1e4 | stress | 1667 | 24 | 0.256637 | 0.666667 | 0.029701 | 0.927277 |
| s3_no_fund_d15_wd01_lr1e4 | vix_rising | 78929 | 316 | 0.336871 | 0.642585 | 0.009593 | 9.827671 |
| s3_no_fund_d15_wd01_lr1e4 | breadth_worsening | 87708 | 357 | 0.304345 | 0.679172 | 0.005404 | 2.674767 |
| s3_no_fund_d25_wd10_lr2e4 | calm | 96330 | 249 | 0.361455 | 0.610687 | 0.001504 | 0.151759 |
| s3_no_fund_d25_wd10_lr2e4 | neutral | 73366 | 414 | 0.291950 | 0.681607 | 0.008204 | 11.168356 |
| s3_no_fund_d25_wd10_lr2e4 | stress | 1667 | 24 | 0.259587 | 0.666667 | 0.027618 | 0.836455 |
| s3_no_fund_d25_wd10_lr2e4 | vix_rising | 78929 | 316 | 0.344659 | 0.639916 | 0.009447 | 9.334936 |
| s3_no_fund_d25_wd10_lr2e4 | breadth_worsening | 87708 | 357 | 0.313725 | 0.675291 | 0.005204 | 2.382917 |
| s3_no_fund_d25_wd05_lr1e4 | calm | 96330 | 249 | 0.390660 | 0.581760 | 0.001166 | 0.076342 |
| s3_no_fund_d25_wd05_lr1e4 | neutral | 73366 | 414 | 0.307511 | 0.651250 | 0.008350 | 12.014478 |
| s3_no_fund_d25_wd05_lr1e4 | stress | 1667 | 24 | 0.286136 | 0.636364 | 0.029017 | 0.893719 |
| s3_no_fund_d25_wd05_lr1e4 | vix_rising | 78929 | 316 | 0.367330 | 0.612848 | 0.009654 | 10.062238 |
| s3_no_fund_d25_wd05_lr1e4 | breadth_worsening | 87708 | 357 | 0.335029 | 0.644058 | 0.005318 | 2.520692 |
| s3_pvv_p16_d35_wd10_lr5e5 | calm | 96330 | 249 | 0.414138 | 0.555444 | 0.001158 | 0.130821 |
| s3_pvv_p16_d35_wd10_lr5e5 | neutral | 73366 | 414 | 0.334187 | 0.619107 | 0.013519 | 129.994608 |
| s3_pvv_p16_d35_wd10_lr5e5 | stress | 1667 | 24 | 0.253687 | 0.651515 | 0.039483 | 1.418154 |
| s3_pvv_p16_d35_wd10_lr5e5 | vix_rising | 78929 | 316 | 0.393205 | 0.576630 | 0.013972 | 49.422245 |
| s3_pvv_p16_d35_wd10_lr5e5 | breadth_worsening | 87708 | 357 | 0.358309 | 0.607466 | 0.008710 | 12.106733 |
| s3_pvv_p16_d25_wd05_lr1e4 | calm | 96330 | 249 | 0.422033 | 0.549618 | 0.001202 | 0.144901 |
| s3_pvv_p16_d25_wd05_lr1e4 | neutral | 73366 | 414 | 0.344830 | 0.608750 | 0.013477 | 126.558110 |
| s3_pvv_p16_d25_wd05_lr1e4 | stress | 1667 | 24 | 0.265487 | 0.651515 | 0.038049 | 1.342167 |
| s3_pvv_p16_d25_wd05_lr1e4 | vix_rising | 78929 | 316 | 0.403442 | 0.568242 | 0.014047 | 50.476344 |
| s3_pvv_p16_d25_wd05_lr1e4 | breadth_worsening | 87708 | 357 | 0.369215 | 0.598226 | 0.008827 | 12.757152 |
| s3_pvv_p16_d15_wd01_lr1e4 | calm | 96330 | 249 | 0.416718 | 0.552832 | 0.000987 | 0.088276 |
| s3_pvv_p16_d15_wd01_lr1e4 | neutral | 73366 | 414 | 0.337959 | 0.614464 | 0.012134 | 71.021088 |
| s3_pvv_p16_d15_wd01_lr1e4 | stress | 1667 | 24 | 0.259587 | 0.651515 | 0.038476 | 1.355105 |
| s3_pvv_p16_d15_wd01_lr1e4 | vix_rising | 78929 | 316 | 0.397350 | 0.574342 | 0.013149 | 37.583624 |
| s3_pvv_p16_d15_wd01_lr1e4 | breadth_worsening | 87708 | 357 | 0.361926 | 0.602292 | 0.007765 | 8.278409 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | calm | 96330 | 249 | 0.434520 | 0.536963 | 0.001552 | 0.248139 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | neutral | 73366 | 414 | 0.356484 | 0.596964 | 0.010131 | 29.036134 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | stress | 1667 | 24 | 0.277286 | 0.636364 | 0.036556 | 1.250573 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | vix_rising | 78929 | 316 | 0.415814 | 0.555852 | 0.011611 | 21.310849 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | breadth_worsening | 87708 | 357 | 0.378539 | 0.588801 | 0.006180 | 3.999634 |
| s3_no_fund_d35_wd10_lr5e5 | calm | 96330 | 249 | 0.449432 | 0.520490 | 0.001696 | 0.291891 |
| s3_no_fund_d35_wd10_lr5e5 | neutral | 73366 | 414 | 0.368676 | 0.583214 | 0.008156 | 11.903356 |
| s3_no_fund_d35_wd10_lr5e5 | stress | 1667 | 24 | 0.297935 | 0.621212 | 0.035882 | 1.222923 |
| s3_no_fund_d35_wd10_lr5e5 | vix_rising | 78929 | 316 | 0.429944 | 0.541555 | 0.009430 | 10.045889 |
| s3_no_fund_d35_wd10_lr5e5 | breadth_worsening | 87708 | 357 | 0.391366 | 0.572353 | 0.004616 | 1.822365 |
| cp148_s2_patchtst_pvv_seq180 | calm | 96587 | 249 | 0.531334 | 0.453552 | 0.002497 | 0.397443 |
| cp148_s2_patchtst_pvv_seq180 | neutral | 76602 | 436 | 0.400516 | 0.559666 | 0.006633 | 6.101107 |
| cp148_s2_patchtst_pvv_seq180 | stress | 2982 | 27 | 0.417763 | 0.533333 | 0.023194 | 0.747064 |
| cp148_s2_patchtst_pvv_seq180 | vix_rising | 81326 | 330 | 0.480875 | 0.514031 | 0.010358 | 14.980100 |
| cp148_s2_patchtst_pvv_seq180 | breadth_worsening | 95019 | 376 | 0.437236 | 0.562461 | 0.005417 | 3.011905 |
| cp148_s2_cnn_lstm_pvv | calm | 96330 | 249 | 0.075335 | 0.922258 | 0.001203 | -0.074041 |
| cp148_s2_cnn_lstm_pvv | neutral | 73366 | 414 | 0.140923 | 0.835179 | -0.000117 | -0.625505 |
| cp148_s2_cnn_lstm_pvv | stress | 1667 | 24 | 0.238938 | 0.772727 | -0.031857 | -0.580901 |
| cp148_s2_cnn_lstm_pvv | vix_rising | 78929 | 316 | 0.110406 | 0.864468 | -0.003820 | -0.856027 |
| cp148_s2_cnn_lstm_pvv | breadth_worsening | 87708 | 357 | 0.122055 | 0.846609 | -0.004412 | -0.903025 |

## 6. horizon bucket별 지표표

| 후보 | bucket | FS | severe | spread | fee |
| --- | --- | --- | --- | --- | --- |
| cp148_s2_patchtst_no_fund_p32_s16 | h1 | 0.321443 | 0.641115 | 0.001862 | 0.310274 |
| cp148_s2_patchtst_no_fund_p32_s16 | h2_h3 | 0.305607 | 0.668024 | 0.004325 | 3.649179 |
| cp148_s2_patchtst_no_fund_p32_s16 | h4_h5 | 0.294381 | 0.693795 | 0.007408 | 14.993380 |
| cp148_s2_patchtst_pvv_p32_s16 | h1 | 0.319013 | 0.641812 | 0.002303 | 0.823952 |
| cp148_s2_patchtst_pvv_p32_s16 | h2_h3 | 0.312175 | 0.669200 | 0.004858 | 5.848699 |
| cp148_s2_patchtst_pvv_p32_s16 | h4_h5 | 0.302974 | 0.684027 | 0.007582 | 17.644306 |
| cp148_s2_tide_pvv | h1 | 0.206388 | 0.776771 | 0.000031 | -0.874534 |
| cp148_s2_tide_pvv | h2_h3 | 0.305462 | 0.711636 | 0.002724 | -0.109675 |
| cp148_s2_tide_pvv | h4_h5 | 0.345967 | 0.654317 | 0.004870 | 1.763433 |
| cp148_s2_patchtst_pvv_p16_s8 | h1 | 0.326853 | 0.650174 | 0.001998 | 0.518287 |
| cp148_s2_patchtst_pvv_p16_s8 | h2_h3 | 0.319031 | 0.667481 | 0.006217 | 17.062477 |
| cp148_s2_patchtst_pvv_p16_s8 | h4_h5 | 0.309079 | 0.681041 | 0.009273 | 65.024719 |
| s3_no_fund_d15_wd01_lr1e4 | h1 | 0.336313 | 0.602555 | 0.002223 | 1.332976 |
| s3_no_fund_d15_wd01_lr1e4 | h2_h3 | 0.326698 | 0.647575 | 0.005900 | 17.783853 |
| s3_no_fund_d15_wd01_lr1e4 | h4_h5 | 0.320073 | 0.664490 | 0.009760 | 116.364861 |
| s3_no_fund_d25_wd10_lr2e4 | h1 | 0.342244 | 0.606039 | 0.002044 | 1.041232 |
| s3_no_fund_d25_wd10_lr2e4 | h2_h3 | 0.338445 | 0.639432 | 0.005590 | 14.101092 |
| s3_no_fund_d25_wd10_lr2e4 | h4_h5 | 0.322098 | 0.664743 | 0.009259 | 84.597871 |
| s3_no_fund_d25_wd05_lr1e4 | h1 | 0.365012 | 0.575145 | 0.002135 | 1.333095 |
| s3_no_fund_d25_wd05_lr1e4 | h2_h3 | 0.359276 | 0.615273 | 0.005807 | 17.504363 |
| s3_no_fund_d25_wd05_lr1e4 | h4_h5 | 0.348050 | 0.636249 | 0.009340 | 88.426525 |
| s3_pvv_p16_d35_wd10_lr5e5 | h1 | 0.361048 | 0.581882 | 0.003337 | 5.361474 |
| s3_pvv_p16_d35_wd10_lr5e5 | h2_h3 | 0.382855 | 0.588129 | 0.008444 | 133.882236 |
| s3_pvv_p16_d35_wd10_lr5e5 | h4_h5 | 0.385748 | 0.589027 | 0.014882 | 5191.120907 |
| s3_pvv_p16_d25_wd05_lr1e4 | h1 | 0.373459 | 0.574216 | 0.003297 | 5.092971 |
| s3_pvv_p16_d25_wd05_lr1e4 | h2_h3 | 0.384938 | 0.587495 | 0.008288 | 119.254208 |
| s3_pvv_p16_d25_wd05_lr1e4 | h4_h5 | 0.400012 | 0.576273 | 0.014771 | 4804.535051 |
| s3_pvv_p16_d15_wd01_lr1e4 | h1 | 0.344298 | 0.596051 | 0.003021 | 4.056122 |
| s3_pvv_p16_d15_wd01_lr1e4 | h2_h3 | 0.355370 | 0.612378 | 0.007338 | 60.823716 |
| s3_pvv_p16_d15_wd01_lr1e4 | h4_h5 | 0.430969 | 0.543729 | 0.014436 | 3729.006910 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | h1 | 0.384916 | 0.559350 | 0.002716 | 3.078592 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | h2_h3 | 0.389423 | 0.580800 | 0.006950 | 44.354650 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | h4_h5 | 0.419396 | 0.557091 | 0.011492 | 425.832678 |
| s3_no_fund_d35_wd10_lr5e5 | h1 | 0.401724 | 0.545645 | 0.002260 | 1.944529 |
| s3_no_fund_d35_wd10_lr5e5 | h2_h3 | 0.408228 | 0.563699 | 0.005817 | 19.512684 |
| s3_no_fund_d35_wd10_lr5e5 | h4_h5 | 0.429696 | 0.545551 | 0.011185 | 373.051783 |
| cp148_s2_patchtst_pvv_seq180 | h1 | 0.472936 | 0.491678 | 0.002243 | 1.068183 |
| cp148_s2_patchtst_pvv_seq180 | h2_h3 | 0.469643 | 0.516397 | 0.005065 | 9.754623 |
| cp148_s2_patchtst_pvv_seq180 | h4_h5 | 0.471950 | 0.524035 | 0.008905 | 80.482550 |
| cp148_s2_cnn_lstm_pvv | h1 | 0.029509 | 0.944715 | -0.001607 | -0.886907 |
| cp148_s2_cnn_lstm_pvv | h2_h3 | 0.090615 | 0.891151 | -0.000865 | -0.854251 |
| cp148_s2_cnn_lstm_pvv | h4_h5 | 0.142865 | 0.847151 | -0.000344 | -0.874138 |

## 7. primary_stage4_base 선정

- primary_stage4_base: `cp148_s2_patchtst_no_fund_p32_s16`
- 이유: line_gate=True, fee>0, false_safe/severe가 existing product 기준보다 개선됐고 Stage 2 best risk 기준을 동시에 만족했다.
- validation false_safe=0.308661, severe=0.685019, spread=0.008389, fee=22.429806
- secondary_stage4_base: `cp148_s2_patchtst_pvv_p32_s16`

## 8. A/B/C/D 실험 방향

- Stage 4 A/B/C/D는 기존처럼 no_fund p32/s16을 주 베이스로 진행한다.
- 단, pvv p16/s8은 기본 보조가 아니라 stress_risk_candidate로 내리고, pvv p32/s16을 secondary_stage4_base로 올린다.
- CNN-LSTM은 LM 제품 후보가 아니라 risk_only_reference로 BM/risk 보조 해석에만 보관한다.
- Stage 3 sweep 후보는 새 risk-aware 기준에서도 Stage 2 best를 넘지 못했으므로 Stage 4 base로 쓰지 않는다.

## 9. product 저장 금지 준수

- product save 없음.
- DB write 없음.
- inference 저장 없음.
- live fetch 없음.
- band/composite 실험 없음.

## 10. python/pythonw 프로세스 확인

- 스크립트 내부 확인은 실행 중인 자기 자신이 잡힐 수 있어 `deferred_external_check`로 기록했다.
- 최종 외부 검증 결과: `Get-Process python,pythonw` 출력 없음.
- 최종 CUDA 확인 결과: visible CUDA compute process 중 Python 학습 프로세스 없음.

## 11. 산출물

- metrics: `docs\cp148_lm_1d_stage4_0_risk_aware_rerank_metrics.json`
- summary csv: `docs\cp148_lm_1d_stage4_0_risk_aware_rerank_summary.csv`
- script: `ai/cp148_lm_1d_stage4_0_risk_aware_rerank.py`
