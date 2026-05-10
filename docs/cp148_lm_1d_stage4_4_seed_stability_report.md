# CP148-LM-1D Stage 4-4 seed stability 보고서

- 한 줄 결론: Stage 4-5 walk-forward 후보 없음
- 생성 시각: 2026-05-10T19:20:23
- 목적: Stage 4-3 Optuna에서 뽑힌 C feature 기반 후보 2개가 단일 seed 우연인지 확인한다.
- 범위: EODHD 500 local parquet, 1D h5, PatchTST line_model, C feature pack, seed 42/7/123
- 금지 준수: product save-run 없음, DB write 없음, inference 저장 없음, live fetch 없음, band/composite 실험 없음
- lambda_direction 정정: artifact에는 기록하지만 PatchTST direction head가 비활성이므로 성능 원인으로 해석하지 않는다.

## 1. 후보별 설정

| 후보 | 역할 | patch | lr | weight_decay | dropout | lambda_direction 해석 |
| --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | Stage 4-3 test 하방 안정성이 가장 좋았던 primary 후보 | 32/16 | 0.0007362816234925851 | 8.143270337695065e-05 | 0.1 | 기록만 함, PatchTST에서는 비활성 해석 |
| trial024_c_risk | Stage 4-3 validation risk 개선폭이 가장 컸던 challenger | 16/8 | 0.0013385598971335333 | 0.000190190440463508 | 0.18 | 기록만 함, PatchTST에서는 비활성 해석 |

## 2. 지표 설명

| 지표 | 방향성 | 의미 |
| --- | --- | --- |
| IC | 높을수록 좋음 | 날짜별 종목 순위 상관이다. |
| spread | 높을수록 좋음 | 예측 상위 10% 실제 수익률에서 예측 하위 10% 실제 수익률을 뺀 값이다. |
| fee_adjusted_return | 높을수록 좋지만 보조 지표 | 10bp 비용 가정 후 ranking signal이 비용을 견디는지 보는 보조 지표다. |
| false_safe_tail_rate | 낮을수록 좋음 | 실제 하위 꼬리인데 line이 0 이상으로 안전하다고 오판한 비율이다. |
| severe_downside_recall | 높을수록 좋음 | 심한 하락을 음수 위험으로 잡아낸 비율이다. |
| conservative_bias | 음수면 보수적이나 너무 음수면 수익 희생 확인 | 예측이 실제보다 평균적으로 낮은지 보는 값이다. |
| upside_sacrifice | 낮을수록 좋음 | 실제 상승 종목을 과하게 낮춰 잡는지 보는 값이다. |
| direction_accuracy | 높을수록 좋음 | 부호 방향이 맞은 비율이다. |
| line_gate | 기본 생존 조건 | 단독 제품 통과 기준이 아니다. |

## 3. 비교 기준

| 기준 | split | IC | spread | fee | false_safe | severe |
| --- | --- | --- | --- | --- | --- | --- |
| stage4_2_c_stress_delta_seed_median | validation | 0.048850 | 0.007700 | 19.680000 | 0.281770 | 0.705290 |
| stage4_2_c_stress_delta_seed_median | test | 0.042510 | 0.006030 | 4.910000 | 0.311710 | 0.687080 |
| stage2_stage4_0_no_fund_p32_s16_primary_base | validation | 0.056970 | 0.008390 | 22.430000 | 0.308661 | 0.685019 |
| stage4_3_trial006_c_balanced_single_seed | validation | 0.054400 | 0.009520 | 63.080000 | 0.237300 | 0.749900 |
| stage4_3_trial006_c_balanced_single_seed | test | 0.042200 | 0.005300 | 3.770000 | 0.281700 | 0.718000 |
| stage4_3_trial024_c_risk_single_seed | validation | 0.048100 | 0.008540 | 41.570000 | 0.224900 | 0.763500 |
| stage4_3_trial024_c_risk_single_seed | test | 0.042500 | 0.005810 | 5.140000 | 0.295000 | 0.706500 |

## 4. Validation seed 집계

| 후보 | 판정 | IC med | spread med | fee med | FS med | FS std | severe med | severe std | upside med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | 보류 후보 | 0.053422 | 0.008438 | 38.505983 | 0.247101 | 0.026059 | 0.740575 | 0.023143 | 0.076252 |
| trial024_c_risk | 보류 후보 | 0.048098 | 0.008540 | 43.978430 | 0.230517 | 0.046423 | 0.757176 | 0.044647 | 0.075842 |

## 5. Test seed 집계

| 후보 | 판정 | IC med | spread med | fee med | FS med | FS std | severe med | severe std | upside med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | 보류 후보 | 0.042207 | 0.005298 | 3.765963 | 0.281680 | 0.036568 | 0.717994 | 0.035097 | 0.079222 |
| trial024_c_risk | 보류 후보 | 0.042502 | 0.005861 | 5.135519 | 0.295006 | 0.038533 | 0.706533 | 0.039239 | 0.075544 |

## 6. Seed별 raw metrics

| 후보 | split | seed | line_gate | IC | spread | fee | FS | severe | bias | upside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | validation | 42 | True | 0.053422 | 0.006552 | 7.530381 | 0.296820 | 0.696794 | -0.022685 | 0.065376 |
| trial006_c_balanced | test | 42 | True | 0.044837 | 0.006704 | 6.458434 | 0.352043 | 0.651559 | -0.017570 | 0.065398 |
| trial006_c_balanced | validation | 7 | True | 0.038892 | 0.008438 | 38.505983 | 0.247101 | 0.740575 | -0.033136 | 0.076467 |
| trial006_c_balanced | test | 7 | True | 0.034465 | 0.004774 | 2.690828 | 0.268859 | 0.732031 | -0.031862 | 0.082069 |
| trial006_c_balanced | validation | 123 | True | 0.054433 | 0.009520 | 63.075829 | 0.237292 | 0.749872 | -0.033112 | 0.076252 |
| trial006_c_balanced | test | 123 | True | 0.042207 | 0.005298 | 3.765963 | 0.281680 | 0.717994 | -0.029967 | 0.079222 |
| trial024_c_risk | validation | 42 | True | 0.040571 | 0.010241 | 138.054896 | 0.326080 | 0.665784 | -0.020273 | 0.062658 |
| trial024_c_risk | test | 42 | True | 0.041446 | 0.006389 | 5.593921 | 0.374070 | 0.626448 | -0.014486 | 0.062196 |
| trial024_c_risk | validation | 7 | True | 0.054002 | 0.008386 | 43.978430 | 0.230517 | 0.757176 | -0.032824 | 0.075842 |
| trial024_c_risk | test | 7 | True | 0.044147 | 0.005861 | 4.985508 | 0.289891 | 0.712520 | -0.027723 | 0.076959 |
| trial024_c_risk | validation | 123 | True | 0.048098 | 0.008540 | 41.565349 | 0.224926 | 0.763497 | -0.033330 | 0.076314 |
| trial024_c_risk | test | 123 | True | 0.042502 | 0.005812 | 5.135519 | 0.295006 | 0.706533 | -0.026421 | 0.075544 |

## 7. Horizon / regime median

| 후보 | split | 구간 | FS med | severe med | spread med | fee med |
| --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | validation | h1 | 0.252445 | 0.687573 | 0.001353 | -0.006003 |
| trial006_c_balanced | validation | h2_h3 | 0.251042 | 0.725208 | 0.003599 | 2.462029 |
| trial006_c_balanced | validation | h4_h5 | 0.231831 | 0.752860 | 0.007331 | 20.649986 |
| trial006_c_balanced | validation | stress | 0.171091 | 0.681818 | 0.026746 | 0.786288 |
| trial006_c_balanced | validation | vix_rising | 0.255731 | 0.719596 | 0.008515 | 6.406735 |
| trial006_c_balanced | validation | breadth_worsening | 0.225971 | 0.742377 | 0.003828 | 1.022048 |
| trial006_c_balanced | test | h1 | 0.287778 | 0.701657 | 0.000991 | 0.084321 |
| trial006_c_balanced | test | h2_h3 | 0.276205 | 0.722673 | 0.002648 | 0.882051 |
| trial006_c_balanced | test | h4_h5 | 0.281301 | 0.720108 | 0.004812 | 3.077635 |
| trial006_c_balanced | test | stress | 0.195257 | 0.798620 | 0.007352 | 0.241538 |
| trial006_c_balanced | test | vix_rising | 0.252608 | 0.744494 | 0.005781 | 1.472002 |
| trial006_c_balanced | test | breadth_worsening | 0.271855 | 0.723634 | 0.003146 | 0.572387 |
| trial024_c_risk | validation | h1 | 0.235839 | 0.714750 | 0.002054 | 1.176417 |
| trial024_c_risk | validation | h2_h3 | 0.222052 | 0.756877 | 0.004803 | 9.312521 |
| trial024_c_risk | validation | h4_h5 | 0.229285 | 0.757921 | 0.007909 | 40.016936 |
| trial024_c_risk | validation | stress | 0.197640 | 0.681818 | 0.027387 | 0.823122 |
| trial024_c_risk | validation | vix_rising | 0.237644 | 0.748189 | 0.010385 | 14.024357 |
| trial024_c_risk | validation | breadth_worsening | 0.206476 | 0.780817 | 0.004863 | 2.151118 |
| trial024_c_risk | test | h1 | 0.306859 | 0.690937 | 0.001075 | 0.142144 |
| trial024_c_risk | test | h2_h3 | 0.289205 | 0.712441 | 0.002982 | 1.186862 |
| trial024_c_risk | test | h4_h5 | 0.293172 | 0.708898 | 0.005338 | 4.079465 |
| trial024_c_risk | test | stress | 0.215892 | 0.782936 | 0.007838 | 0.268714 |
| trial024_c_risk | test | vix_rising | 0.263097 | 0.731127 | 0.005956 | 1.619033 |
| trial024_c_risk | test | breadth_worsening | 0.287440 | 0.712140 | 0.003523 | 0.735335 |

## 8. 후보 판단

### trial006_c_balanced

- 판정: 보류 후보
- 이유: median 기준으로는 test false_safe 0.281680, severe 0.717994라 좋아 보이지만 seed 42 test에서 false_safe 0.352043, severe 0.651559로 크게 흔들렸다. false_safe는 실제 위험 꼬리를 안전하다고 보는 비율이므로, 이 정도 seed 흔들림이면 Stage 4-5로 올리기 어렵다.
- Stage 4-3 단일 seed 대비: seed 3개 median을 기준으로 재현성을 확인했다.
- Stage 4-3 단일 seed 대비 변화: validation false_safe는 0.237300에서 0.247101로 악화, validation severe는 0.749900에서 0.740575로 악화했다. test false_safe는 0.281700에서 0.281680으로 거의 유지됐지만, seed별 max가 0.352043까지 올라가 안정성이 부족하다.

### trial024_c_risk

- 판정: 보류 후보
- 이유: median 기준으로는 test false_safe 0.295006, severe 0.706533이라 기준 C보다는 낫지만 seed 42 test에서 false_safe 0.374070, severe 0.626448로 무너졌다. Stage 4-3에서 보였던 강한 validation risk 개선이 seed 3개에서 안정적으로 재현됐다고 보기 어렵다.
- Stage 4-3 단일 seed 대비: seed 3개 median을 기준으로 재현성을 확인했다.
- Stage 4-3 단일 seed 대비 변화: validation false_safe는 0.224900에서 0.230517로 약간 악화, validation severe는 0.763500에서 0.757176으로 약간 악화했다. test false_safe와 severe median은 단일 seed와 거의 같지만 seed별 max/min 차이가 커서 challenger로 보류한다.

## 9. Stage 4-2 / Stage 4-3 대비 해석

- Stage 4-2 C_stress_delta seed median 대비 두 후보 모두 validation/test median 기준 false_safe와 severe는 개선했다. 즉 C feature pack 자체의 하방 신호는 아직 살아 있다.
- 그러나 product save-run이나 walk-forward로 넘기기에는 seed 42 test 결과가 너무 약하다. trial006은 seed 42 test false_safe가 0.352043이고, trial024는 0.374070이다.
- spread는 두 후보 모두 test median 양수다. trial006은 0.005298, trial024는 0.005861로 최소 기준은 넘는다. 다만 이 spread 개선만으로 false_safe 흔들림을 덮으면 안 된다.
- fee_adjusted_return도 둘 다 test median 양수다. trial006은 3.765963, trial024는 5.135519다. 하지만 fee는 보조 지표이므로 false_safe/severe가 흔들리면 단독 채택 근거가 아니다.
- trial006은 상대적으로 더 균형적이고 stress 구간 test false_safe/severe median이 좋다. 그래도 seed 42 전체 test 붕괴 때문에 Stage 4-5 후보로 올리지는 않는다.
- trial024는 validation risk는 강하지만 seed 42에서 가장 크게 흔들렸다. 따라서 challenger 유지도 조심스럽고, 현재는 보류 후보로 기록한다.

## 10. 다음 액션

- 최종 판단: Stage 4-5 walk-forward 후보 없음
- Stage 4-5로 넘어가는 경우 product save-run이 아니라 3-fold walk-forward 검증으로만 진행한다.
- 둘 다 흔들린 경우 product candidate save-run은 금지하고 Stage 4-3 후보 재선정으로 되돌린다.
- 다음 LM은 Stage 4-3으로 되돌아가되, 단일 seed objective보다 seed 42 취약성을 줄이는 기준을 먼저 세우는 편이 낫다. 예를 들어 후보 선택 시 validation median뿐 아니라 seed 42 test-like stress 분해 또는 conservative selector의 seed 민감도를 별도 penalty로 기록한다.
- lambda_direction은 이번에도 artifact 기록값일 뿐이다. PatchTST direction head가 비활성이므로 성능 원인을 lambda_direction 효과라고 말하지 않는다.

## 11. 산출물

- metrics: `docs\cp148_lm_1d_stage4_4_seed_stability_metrics.json`
- summary: `docs\cp148_lm_1d_stage4_4_seed_stability_summary.csv`
- logs/meta: `logs\cp148_lm_1d_stage4_4_seed_stability`
- script: `ai/cp148_lm_1d_stage4_4_seed_stability.py`
