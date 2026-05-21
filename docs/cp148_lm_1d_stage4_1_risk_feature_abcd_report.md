# CP148-LM-1D Stage 4-1 risk feature A/B/C/D 보고서

- 생성 시각: 2026-05-09T11:56:14
- 범위: PatchTST no_fund p32/s16 기반 risk-aware selector 및 실험용 피처 확장
- 금지 준수: save-run 없음, DB write 없음, inference 저장 없음, product promotion 없음, live fetch 없음, band/composite 없음
- CNN-LSTM은 이번 LM 실험에서 제외하고 risk_only_reference로만 보관

## 1. 전체 validation 결과

| 실험 | 분류 | line_gate | IC | spread | fee | false_safe | severe | bias | sacrifice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_selector_only | stress_risk_improvement | True | 0.051346 | 0.007703 | 14.884894 | 0.303391 | 0.689162 | -0.022171 | 0.064499 |
| B_atr_only | stress_risk_improvement | True | 0.056811 | 0.007380 | 16.004600 | 0.294363 | 0.700495 | -0.023286 | 0.066412 |
| C_stress_delta | strong_stage4_candidate | True | 0.058450 | 0.008607 | 30.758173 | 0.295834 | 0.698346 | -0.022547 | 0.065223 |
| D_stock_fragility | strong_stage4_candidate | True | 0.049825 | 0.008396 | 29.049106 | 0.278216 | 0.710134 | -0.026191 | 0.068749 |

## 2. stress regime 결과

| 실험 | 구간 | samples | dates | FS | severe | spread | fee |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_selector_only | calm | 96330 | 249 | 0.313622 | 0.666131 | 0.000989 | -0.061257 |
| A_selector_only | neutral | 73366 | 414 | 0.275850 | 0.700179 | 0.004612 | 1.254577 |
| A_selector_only | stress | 1667 | 24 | 0.203540 | 0.636364 | 0.029727 | 0.921702 |
| A_selector_only | vix_rising | 78929 | 316 | 0.305156 | 0.684140 | 0.008002 | 4.566645 |
| A_selector_only | breadth_worsening | 87708 | 357 | 0.282647 | 0.705600 | 0.002926 | 0.305357 |
| B_atr_only | calm | 96330 | 249 | 0.300826 | 0.682603 | 0.002010 | 0.231790 |
| B_atr_only | neutral | 73366 | 414 | 0.271068 | 0.710536 | 0.005669 | 3.044661 |
| B_atr_only | stress | 1667 | 24 | 0.209440 | 0.666667 | 0.011777 | 0.264066 |
| B_atr_only | vix_rising | 78929 | 316 | 0.295924 | 0.695196 | 0.007301 | 4.054682 |
| B_atr_only | breadth_worsening | 87708 | 357 | 0.275018 | 0.721678 | 0.002449 | 0.213109 |
| C_stress_delta | calm | 96330 | 249 | 0.298297 | 0.684612 | 0.001410 | 0.032446 |
| C_stress_delta | neutral | 73366 | 414 | 0.281374 | 0.706250 | 0.006766 | 4.477954 |
| C_stress_delta | stress | 1667 | 24 | 0.221239 | 0.651515 | 0.023921 | 0.673890 |
| C_stress_delta | vix_rising | 78929 | 316 | 0.298248 | 0.697674 | 0.009205 | 7.011952 |
| C_stress_delta | breadth_worsening | 87708 | 357 | 0.283099 | 0.714101 | 0.003630 | 0.628743 |
| D_stock_fragility | calm | 96330 | 249 | 0.302838 | 0.674769 | 0.000688 | -0.079783 |
| D_stock_fragility | neutral | 73366 | 414 | 0.237791 | 0.731250 | 0.006936 | 5.889924 |
| D_stock_fragility | stress | 1667 | 24 | 0.238938 | 0.666667 | 0.029307 | 0.905539 |
| D_stock_fragility | vix_rising | 78929 | 316 | 0.288890 | 0.699390 | 0.009693 | 9.981272 |
| D_stock_fragility | breadth_worsening | 87708 | 357 | 0.257558 | 0.731288 | 0.003739 | 0.902983 |

## 3. horizon bucket 결과

| 실험 | bucket | FS | severe | spread | fee |
| --- | --- | --- | --- | --- | --- |
| A_selector_only | h1 | 0.306631 | 0.650639 | 0.001098 | -0.188996 |
| A_selector_only | h2_h3 | 0.301962 | 0.671734 | 0.003569 | 1.870749 |
| A_selector_only | h4_h5 | 0.290302 | 0.697338 | 0.007302 | 15.530772 |
| B_atr_only | h1 | 0.303506 | 0.652265 | 0.001427 | 0.132960 |
| B_atr_only | h2_h3 | 0.291459 | 0.688111 | 0.003774 | 2.897696 |
| B_atr_only | h4_h5 | 0.281651 | 0.708776 | 0.007201 | 17.854667 |
| C_stress_delta | h1 | 0.307152 | 0.664111 | 0.001514 | 0.061280 |
| C_stress_delta | h2_h3 | 0.294497 | 0.686573 | 0.004475 | 4.221658 |
| C_stress_delta | h4_h5 | 0.284342 | 0.704272 | 0.008122 | 28.553526 |
| D_stock_fragility | h1 | 0.286582 | 0.662253 | 0.001625 | 0.448276 |
| D_stock_fragility | h2_h3 | 0.277225 | 0.701773 | 0.004714 | 6.885082 |
| D_stock_fragility | h4_h5 | 0.268632 | 0.718595 | 0.007771 | 25.683316 |

## 4. 결론

- 가장 낮은 전체 false_safe 후보: `D_stock_fragility`
- A_selector_only는 Stage 2 primary 대비 false_safe 0.308661 -> 0.303391, severe 0.685019 -> 0.689162로 소폭 개선했다. 다만 spread가 0.007703으로 0.008 권장선 아래라 selector만으로는 충분한 해결책이 아니다.
- B_atr_only는 false_safe 0.294363, severe 0.700495까지 개선했고 h1 false_safe도 0.303506으로 A보다 낮았다. ATR은 단기 불안정성 인지에 도움을 줬지만 spread 0.007380으로 ranking 희생이 있었다.
- C_stress_delta는 IC 0.058450, spread 0.008607, fee 30.758173으로 ranking/fee를 가장 잘 살린 strong 후보였다. 그러나 stress false_safe는 0.221239로 A/B보다 높아, stress delta가 stress 구간 오판을 직접 줄였다고 보기는 어렵다.
- D_stock_fragility는 false_safe 0.278216, severe 0.710134로 전체 하방 안정성이 가장 좋고, h1/h2_h3/h4_h5 모든 bucket에서 false_safe를 가장 낮췄다. 개별 종목의 drawdown/downside volatility가 이번 질문에서 가장 큰 개선 원인으로 보인다.
- primary_stage4_base는 기존 no_fund p32/s16 원형에서 `D_stock_fragility` 변형으로 옮겨 seed 3개 재평가 후보로 둔다. `C_stress_delta`는 alpha를 가장 잘 보존한 비교 후보로 같이 보관한다.
- secondary_stage4_base인 pvv p32/s16에 같은 A/B/C/D를 바로 확장하기보다는, 먼저 D/C의 seed 안정성을 확인한 뒤 필요할 때 확장한다.
- CNN-LSTM은 이번 LM 실험에서 제외했고, spread/fee 실패 때문에 line base가 아니라 risk_only_reference로만 보관한다.
- 터미널 실행 주의: PowerShell `Start-Process`가 `Path/PATH` 환경변수 중복 때문에 실패할 수 있다. 장시간 실행은 `.venv\Scripts\python.exe` 직접 실행 또는 환경변수 정리 후 로그 파일을 붙여 실행해야 한다.

## 5. 산출물

- design: `docs\cp148_lm_1d_stage4_revised_experiment_design.md`
- metrics: `docs\cp148_lm_1d_stage4_1_risk_feature_abcd_metrics.json`
- summary: `docs\cp148_lm_1d_stage4_1_risk_feature_abcd_summary.csv`
- script: `ai/cp148_lm_1d_stage4_1_risk_feature_abcd.py`
