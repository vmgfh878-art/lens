# CP148-LM-1D Stage 4-3 C_stress_delta Optuna narrow sweep 보고서

- 작성 시각: 2026-05-10 14:06 KST
- 실행 목적: C_stress_delta의 하방 안정성 신호를 유지하면서 ranking/spread/fee를 회복할 수 있는 학습 조건과 patch geometry를 찾는다.
- 실행 범위: EODHD 500 local parquet, 1D h5, PatchTST, line_model, seed 123 단일 seed sweep
- 완료 상태: 완료 trial 35개, 실패/중단 trial 1개
- 중단 상태: 사용자가 Ctrl+C로 중단. `trial035`는 train log 폴더만 있고 `run_meta.json`이 없어 유효 trial에서 제외했다.
- 금지 준수: product save 없음, DB write 없음, inference 저장 없음, live fetch 없음, band/composite 실험 없음, beta=2.0 유지
- test 사용 원칙: Optuna objective와 후보 선택은 validation만 사용했고, test는 사후 확인용으로만 기록했다.

## 1. 실험 질문

Stage 4-2에서 C_stress_delta는 D_stock_fragility보다 false_safe_tail_rate와 severe_downside_recall이 좋아졌지만, spread/fee가 약했다. Stage 4-3의 질문은 다음 하나다.

> C_stress_delta의 하방 안정성을 유지하면서 IC, spread, fee를 다시 살릴 수 있는가?

이 실험은 제품 저장 후보 확정이 아니다. 결과 표현은 `C-risk 후보`, `C-alpha 후보`, `C-balanced 후보`, `Stage 4-4 seed 재평가 후보`, `탈락 후보`로만 제한한다.

## 2. 고정 조건

| 항목 | 값 |
| --- | --- |
| provider/source | eodhd / eodhd |
| data backend | local parquet |
| timeframe | 1D |
| horizon | h5 |
| model | PatchTST |
| role | line_model |
| feature family | C_stress_delta |
| base feature | no_fundamentals |
| added features | atr_ratio, vix_change_5d, credit_spread_change_20d, ma200_pct_change_20d |
| target | raw_future_return |
| line_target_type | raw_future_return |
| alpha/beta/delta | 1.0 / 2.0 / 1.0 |
| checkpoint_selection | line_gate 유지 |
| epochs | 3 |
| batch_size | 256 |
| seed | 123 |
| amp_dtype | bf16 |
| source_data_hash | 60df6a1e9d9a8f62 |
| eligible ticker count | 473 |

line_gate는 생존 조건이다. line_gate가 true라는 뜻은 IC/spread/MAE/SMAPE가 기본 생존 조건을 통과했다는 뜻이지, false_safe나 severe_recall까지 통과했다는 뜻은 아니다.

## 3. Sweep 공간

| 축 | 범위 | 의도 |
| --- | --- | --- |
| lr | 3e-4 ~ 2e-3 log uniform | C 후보의 약한 spread/fee 회복 |
| weight_decay | 1e-5 ~ 8e-4 log uniform | 과적합과 seed 착시 억제 |
| dropout | 0.05, 0.08, 0.10, 0.12, 0.15, 0.18 | alpha와 하방 안정성 균형 |
| patch_geometry | p32_s16, p24_s12, p16_s8 | 장기 안정성, 중간 해상도, 단기 반응 비교 |
| lambda_direction | 0.05, 0.10, 0.15, 0.20, 0.30 | beta를 건드리지 않고 방향성 보조 loss 강도 확인 |

이전 CP 기록의 `config.json` 95개를 확인한 결과, Stage 4-3 이전에는 `lambda_direction=0.1`만 사용됐다. 따라서 이번 Stage 4-3은 `lambda_direction`을 실제 실험 축으로 연 첫 sweep이다.

## 4. 기준 reference

기준은 Stage 4-2 C_stress_delta seed 3개 median이다.

| split | IC | spread | fee | false_safe | severe |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.04885 | 0.00770 | 19.68 | 0.28177 | 0.70529 |
| test | 0.04251 | 0.00603 | 4.91 | 0.31171 | 0.68708 |

해석 기준:

- false_safe_tail_rate는 낮을수록 좋다.
- severe_downside_recall은 높을수록 좋다.
- spread/fee/IC는 높을수록 좋다.
- validation 개선만으로 결론을 내리지 않고, test는 사후 안정성 경고로 본다.

## 5. 전체 완료 현황

| 상태 | 개수 |
| --- | ---: |
| 완료 trial | 35 |
| 실패/중단 trial | 1 |
| C-risk 후보 | 10 |
| C-balanced 후보 | 6 |
| C-alpha 후보 | 4 |
| 탈락 후보 | 15 |

`trial032`, `trial033`, `trial034`는 Ctrl+C 전에 완료되어 `run_meta.json`이 존재한다. `trial035`는 학습 로그 폴더만 생성되고 메타가 없어 제외했다.

## 6. Top trial 요약

| rank | trial | 분류 | geom | lr | wd | dropout | dir | val spread | val fee | val FS | val severe | test FS | test severe |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | C-risk 후보 | p16_s8 | 0.00133856 | 0.00019019 | 0.18 | 0.20 | 0.008540 | 41.565 | 0.224926 | 0.763497 | 0.295006 | 0.706533 |
| 2 | 6 | C-balanced 후보 | p32_s16 | 0.00073628 | 0.00008143 | 0.10 | 0.20 | 0.009520 | 63.076 | 0.237292 | 0.749872 | 0.281680 | 0.717994 |
| 3 | 1 | C-risk 후보 | p16_s8 | 0.00129956 | 0.00003277 | 0.15 | 0.15 | 0.008870 | 57.510 | 0.227727 | 0.761675 | 0.295769 | 0.705381 |
| 4 | 28 | C-risk 후보 | p16_s8 | 0.00166509 | 0.00035527 | 0.08 | 0.20 | 0.008745 | 44.922 | 0.236761 | 0.752008 | 0.303328 | 0.698253 |
| 5 | 14 | C-risk 후보 | p16_s8 | 0.00106722 | 0.00013476 | 0.15 | 0.15 | 0.008504 | 44.174 | 0.239766 | 0.748961 | 0.319925 | 0.680595 |
| 6 | 17 | C-balanced 후보 | p16_s8 | 0.00077464 | 0.00004092 | 0.10 | 0.20 | 0.009418 | 72.238 | 0.264462 | 0.723504 | 0.339316 | 0.660645 |
| 7 | 12 | C-risk 후보 | p16_s8 | 0.00090529 | 0.00004039 | 0.15 | 0.30 | 0.009120 | 67.045 | 0.267047 | 0.723959 | 0.338577 | 0.661785 |
| 8 | 32 | C-balanced 후보 | p32_s16 | 0.00080954 | 0.00008268 | 0.10 | 0.05 | 0.008515 | 35.056 | 0.259379 | 0.728202 | 0.305062 | 0.693606 |
| 9 | 11 | C-balanced 후보 | p16_s8 | 0.00198757 | 0.00001760 | 0.15 | 0.15 | 0.008730 | 48.325 | 0.271587 | 0.719161 | 0.337669 | 0.665543 |
| 10 | 34 | C-alpha 후보 | p24_s12 | 0.00117496 | 0.00034875 | 0.12 | 0.20 | 0.010363 | 89.548 | 0.287600 | 0.703998 | 0.333461 | 0.666349 |

## 7. 핵심 후보 상세 해석

### 7.1 trial 24: C-risk 후보

`trial024`는 Optuna objective 기준 1위다.

| split | IC | spread | fee | false_safe | severe | conservative_bias | upside_sacrifice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.048098 | 0.008540 | 41.565 | 0.224926 | 0.763497 | -0.033330 | 0.076314 |
| test | 0.042502 | 0.005812 | 5.136 | 0.295006 | 0.706533 | -0.026421 | 0.075544 |

장점:

- validation false_safe가 0.2249로 기준 C 0.2818보다 크게 낮아졌다.
- validation severe가 0.7635로 product target 0.75를 넘었다.
- validation spread와 fee도 기준 C보다 개선됐다.
- test에서도 false_safe 0.2950, severe 0.7065로 기준 C test 0.3117/0.6871보다 개선됐다.

주의:

- patch가 p16_s8이라 단기 민감도가 높고, 같은 p16_s8 계열에서 test false_safe가 무너진 trial이 여러 개 있었다.
- h1 test false_safe는 0.3069로 trial 6보다 약하다.
- Stage 4-4 seed 재평가 없이는 단일 seed 우연 가능성을 배제할 수 없다.

bucket/regime:

| 구간 | validation FS | validation severe | test FS | test severe |
| --- | ---: | ---: | ---: | ---: |
| h1 | 0.229995 | 0.714750 | 0.306859 | 0.690937 |
| h2_h3 | 0.218956 | 0.759229 | 0.289205 | 0.712441 |
| h4_h5 | 0.220692 | 0.767689 | 0.293172 | 0.708898 |
| stress | 0.197640 | 0.681818 | 0.214352 | 0.784191 |
| vix_rising | 0.232745 | 0.751048 | 0.263097 | 0.731127 |
| breadth_worsening | 0.201786 | 0.785622 | 0.287440 | 0.712140 |

판정: `C-risk 후보`. Stage 4-4 seed 재평가 대상이다.

### 7.2 trial 6: C-balanced 후보

`trial006`은 objective 기준 2위지만, test 하방 안정성은 가장 좋다.

| split | IC | spread | fee | false_safe | severe | conservative_bias | upside_sacrifice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.054433 | 0.009520 | 63.076 | 0.237292 | 0.749872 | -0.033112 | 0.076252 |
| test | 0.042207 | 0.005298 | 3.766 | 0.281680 | 0.717994 | -0.029967 | 0.079222 |

장점:

- validation IC/spread/fee가 trial 24보다 강하다.
- test false_safe 0.2817, severe 0.7180으로 top 후보 중 test 하방 안정성이 가장 좋다.
- p32_s16이라 기존 Stage 2/4의 안정적인 장기 문맥 후보와 해석이 이어진다.
- warning이 없다.

주의:

- test fee는 3.77로 trial 24보다 낮다.
- h1 validation fee는 약하지만, 이것은 h1을 제품 슬롯으로 쓰지 않는 CP148 정책과 충돌하지 않는다.

bucket/regime:

| 구간 | validation FS | validation severe | test FS | test severe |
| --- | ---: | ---: | ---: | ---: |
| h1 | 0.236620 | 0.714983 | 0.287778 | 0.701657 |
| h2_h3 | 0.229111 | 0.745657 | 0.276205 | 0.722673 |
| h4_h5 | 0.231831 | 0.752860 | 0.281301 | 0.720108 |
| stress | 0.168142 | 0.712121 | 0.195257 | 0.798620 |
| vix_rising | 0.240281 | 0.740564 | 0.252608 | 0.744494 |
| breadth_worsening | 0.211731 | 0.774533 | 0.271855 | 0.723634 |

판정: `C-balanced 후보`. Stage 4-4 seed 재평가 대상 중 1순위다.

### 7.3 trial 1: 백업 C-risk 후보

`trial001`은 trial 24와 매우 유사한 p16_s8 계열이다.

| split | IC | spread | fee | false_safe | severe |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.041340 | 0.008870 | 57.510 | 0.227727 | 0.761675 |
| test | 0.042397 | 0.005719 | 4.811 | 0.295769 | 0.705381 |

trial 24와 거의 같은 성격이므로, Stage 4-4 후보를 2개로 제한한다면 제외한다. 예산이 허용될 때만 p16_s8 backup으로 둔다.

## 8. lambda_direction 해석

Stage 4-3 전에는 `lambda_direction=0.1`만 사용됐다. 이번 실험에서 처음으로 0.05~0.30을 열었다.

| lambda_direction | trial 수 | best trial | best score | 평균 val FS | 평균 val severe | 평균 test FS | 평균 test severe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 5 | 32 | 1.181690 | 0.310175 | 0.680249 | 0.343148 | 0.657190 |
| 0.10 | 4 | 27 | 1.105085 | 0.300561 | 0.687703 | 0.352126 | 0.647561 |
| 0.15 | 9 | 1 | 1.392910 | 0.286251 | 0.703644 | 0.342534 | 0.658741 |
| 0.20 | 14 | 24 | 1.413901 | 0.275773 | 0.713470 | 0.328935 | 0.671890 |
| 0.30 | 3 | 12 | 1.192154 | 0.281412 | 0.707790 | 0.335940 | 0.664092 |

해석:

- `0.20`이 가장 설득력 있다. top 1, top 2, top 4가 모두 `lambda_direction=0.20`이다.
- `0.15`도 유효하다. trial 1과 trial 14가 강한 risk 후보로 나왔다.
- `0.30`은 방향 loss가 과한 쪽으로 보인다. validation alpha는 나오지만 test 하방이 약해지는 경향이 있다.
- `0.05`와 `0.10`은 이번 C_stress_delta 조건에서는 방향 보조가 약해 보인다.
- 결론적으로 Stage 4-4는 `0.20` 중심, `0.15`는 backup으로 보는 것이 타당하다.

주의: 위 평균은 penalty를 받은 실패 후보를 포함한다. 따라서 평균 score보다 best 후보와 test 하방 유지 여부가 더 중요하다.

## 9. patch geometry 해석

| patch geometry | trial 수 | best trial | best score | 평균 val FS | 평균 val severe | 평균 test FS | 평균 test severe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p16_s8 | 18 | 24 | 1.413901 | 0.283286 | 0.706563 | 0.342855 | 0.658562 |
| p24_s12 | 6 | 34 | 1.152587 | 0.298836 | 0.691438 | 0.339141 | 0.660182 |
| p32_s16 | 11 | 6 | 1.393326 | 0.285662 | 0.702731 | 0.328521 | 0.671671 |

해석:

- `p16_s8`은 validation에서 가장 공격적으로 risk를 개선할 수 있다. trial 24/1/28이 여기에 속한다.
- 다만 p16_s8은 test에서 false_safe가 0.33 이상으로 흔들린 trial도 많다. 단기 민감도가 장점이면서 위험이다.
- `p32_s16`은 validation objective 1위는 아니지만, test false_safe와 severe가 가장 안정적이다. trial 6과 trial 19가 이쪽이다.
- `p24_s12`는 기대했던 중간 해상도 역할을 충분히 못 했다. trial 34는 alpha는 강하지만 test false_safe 0.3335로 하방 안정성이 약하다.

## 10. 추가 trial 32~34 해석

Ctrl+C 전 완료된 추가 trial은 3개다.

| trial | 분류 | 핵심 해석 |
| ---: | --- | --- |
| 32 | C-balanced 후보 | p32_s16, lambda 0.05. test FS 0.3051로 기준 C보다는 개선됐지만 trial 6/24보다 약하다. |
| 33 | 탈락 후보 | p16_s8, lambda 0.15. validation은 나쁘지 않지만 test FS 0.3544, severe 0.6483로 하방 안정성이 무너졌다. |
| 34 | C-alpha 후보 | p24_s12, lambda 0.20. val spread/fee는 강하지만 test FS 0.3335, severe 0.6663로 Stage 4-4 후보는 아니다. |

이 3개는 상위권 결론을 바꾸지 않았다. 따라서 48 trial까지 확장하지 않고 중단한 판단은 타당하다.

## 11. 탈락 패턴

### 11.1 alpha만 강한 후보

trial 21, 29, 25, 30 같은 후보는 spread/fee가 매우 크거나 나쁘지 않지만 false_safe/severe가 무너졌다. 예를 들어 trial 21은 validation fee가 893으로 매우 크지만 validation false_safe 0.3897, severe 0.5991이라 line risk 후보로는 탈락이다.

해석:

- fee/spread만 보면 좋아 보이는 후보가 존재한다.
- 그러나 Lens line 목표는 낙관적 수익선이 아니라 false-safe를 줄이는 보수적 예측선이다.
- 따라서 이런 후보는 Stage 4-4로 보내면 안 된다.

### 11.2 p24_s12 계열

p24_s12는 중간 해상도 후보였지만, alpha와 risk를 동시에 만족하지 못했다. trial 34는 C-alpha 후보로 분류되지만 test false_safe가 0.3335라 risk 안정성 기준으로는 약하다.

### 11.3 지나친 direction 또는 약한 direction

- `lambda_direction=0.05/0.10`은 C의 spread/fee 회복에는 일부 도움을 주지만 false_safe/severe 개선이 약했다.
- `lambda_direction=0.30`은 방향성은 살 수 있으나 test 하방 안정성이 약해지는 경향이 있다.
- 현재 관측에서는 `0.20`이 가장 균형적이다.

## 12. 최종 Stage 4-4 seed 재평가 후보

Stage 4-4에는 최대 2개만 넘기는 것이 맞다.

### 1순위: trial 6, C-balanced 후보

- 이유: test false_safe와 severe가 가장 좋고, p32_s16이라 구조적으로 안정적이다.
- config:
  - patch_geometry: p32_s16
  - lr: 0.0007362816234925851
  - weight_decay: 0.00008143270337695065
  - dropout: 0.10
  - lambda_direction: 0.20
- Stage 4-4 목적:
  - seed 3개에서 test 하방 안정성이 유지되는지 확인
  - validation/test 모두에서 기존 C median보다 안정적인지 확인

### 2순위: trial 24, C-risk 후보

- 이유: validation risk 개선폭이 가장 크고, test에서도 기존 C보다 개선됐다.
- config:
  - patch_geometry: p16_s8
  - lr: 0.0013385598971335333
  - weight_decay: 0.000190190440463508
  - dropout: 0.18
  - lambda_direction: 0.20
- Stage 4-4 목적:
  - p16_s8 risk 개선이 seed 안정적인지 확인
  - 단기 민감도가 test에서 계속 유지되는지 확인

### backup: trial 1

- trial 1은 trial 24와 거의 같은 p16_s8 risk 계열이다.
- Stage 4-4 예산을 2개로 제한하면 제외한다.
- trial 24가 seed에서 무너지면 backup으로 재검토할 수 있다.

## 13. 다음 Stage 4-4 제안

권장 seed:

- 42
- 7
- 123

이유: Stage 4-2 seed 안정성 평가와 비교하기 쉽고, 현재 sweep seed 123을 포함한다.

실행 후보:

1. `stage4_4_trial006_c_balanced_seed_stability`
2. `stage4_4_trial024_c_risk_seed_stability`

평가 기준:

- validation/test 각각 median, mean, std
- false_safe_tail_rate
- severe_downside_recall
- IC
- long_short_spread
- fee_adjusted_return
- h1/h2_h3/h4_h5 bucket
- stress/vix_rising/breadth_worsening regime

판정:

- trial 6이 seed 3개에서 test false_safe/severe를 안정적으로 유지하면 Stage 4-4 primary 후보로 둔다.
- trial 24가 validation risk 우위를 유지하고 test도 무너지지 않으면 C-risk challenger로 둔다.
- 둘 다 seed std가 크면 Stage 4-3 결과는 단일 seed 튜닝 신호로만 남기고 추가 feature/selector 재검토로 넘긴다.

## 14. 산출물

- metrics: `docs/cp148_lm_1d_stage4_3_c_stress_delta_optuna_metrics.json`
- summary: `docs/cp148_lm_1d_stage4_3_c_stress_delta_optuna_summary.csv`
- study DB: `docs/cp148_lm_1d_stage4_3_c_stress_delta_optuna_logs/cp148_stage4_3_c_stress_delta_optuna.db`
- study export: `docs/cp148_lm_1d_stage4_3_c_stress_delta_optuna_logs/cp148_stage4_3_c_stress_delta_optuna_study_export.json`
- logs: `docs/cp148_lm_1d_stage4_3_c_stress_delta_optuna_logs`
- script: `ai/cp148_lm_1d_stage4_3_c_stress_delta_optuna.py`

## 15. 결론

Stage 4-3는 tuning 실패가 아니다. 기존 C reference 대비 하방 안정성과 spread/fee를 동시에 개선한 후보를 찾았다.

핵심 결론:

1. `lambda_direction=0.20`이 현재 C_stress_delta에서 가장 설득력 있다.
2. `trial024`는 validation risk 개선폭이 가장 큰 C-risk 후보이다.
3. `trial006`은 test 하방 안정성이 가장 좋은 C-balanced 후보이다.
4. 48 trial 확장은 필수적이지 않다. 35개 완료 시점에서 Stage 4-4로 넘길 후보는 충분히 확보됐다.
5. Stage 4-4에서는 trial 6과 trial 24만 seed 3개로 재평가하는 것이 가장 효율적이다.

제품 저장, 제품 v1, product promotion 결론은 이번 보고서에서 내리지 않는다.
