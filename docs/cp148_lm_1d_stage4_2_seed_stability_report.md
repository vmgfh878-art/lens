# CP148-LM-1D Stage 4-2 seed 안정성 보고서

- 생성 시각: 2026-05-09T15:19:02
- 범위: D_stock_fragility와 C_stress_delta의 seed 3개 재평가
- seed: 42, 7, 123
- 금지 준수: product save 없음, DB write 없음, inference 저장 없음, live fetch 없음, band/composite 없음, beta=2 유지
- line_gate 의미는 생존 조건으로 유지했고, checkpoint 정렬만 Stage 4 risk-aware selector를 사용했다.

## 1. Validation median

| 후보 | 판정 | IC | spread | fee | false_safe | FS std | severe | severe std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_stock_fragility | 탈락 후보 | 0.049825 | 0.009699 | 64.752047 | 0.316492 | 0.048154 | 0.677644 | 0.048321 |
| C_stress_delta | Stage 4-2 primary 후보 | 0.048854 | 0.007697 | 19.679063 | 0.281770 | 0.012695 | 0.705294 | 0.011445 |

## 2. Test median

| 후보 | 판정 | IC | spread | fee | false_safe | FS std | severe | severe std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_stock_fragility | 탈락 후보 | 0.039707 | 0.005739 | 4.522556 | 0.360539 | 0.030701 | 0.642139 | 0.030128 |
| C_stress_delta | Stage 4-2 primary 후보 | 0.042514 | 0.006031 | 4.907032 | 0.311708 | 0.024087 | 0.687085 | 0.022684 |

## 3. Seed별 validation/test 요약

| 후보 | split | seed | line_gate | IC | spread | fee | FS | severe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_stock_fragility | validation | 42 | True | 0.049825 | 0.008396 | 29.049106 | 0.278216 | 0.710134 |
| D_stock_fragility | test | 42 | True | 0.039707 | 0.005739 | 4.522556 | 0.314979 | 0.685661 |
| D_stock_fragility | validation | 7 | True | 0.055773 | 0.009699 | 64.752047 | 0.316492 | 0.677644 |
| D_stock_fragility | test | 7 | True | 0.046317 | 0.006833 | 6.580137 | 0.360539 | 0.642139 |
| D_stock_fragility | validation | 123 | True | 0.039358 | 0.010414 | 208.349020 | 0.393977 | 0.595321 |
| D_stock_fragility | test | 123 | True | 0.031184 | 0.004988 | 3.934571 | 0.389573 | 0.612286 |
| C_stress_delta | validation | 42 | True | 0.058450 | 0.008607 | 30.758173 | 0.295834 | 0.698346 |
| C_stress_delta | test | 42 | True | 0.046223 | 0.006756 | 6.333854 | 0.356355 | 0.647697 |
| C_stress_delta | validation | 7 | True | 0.044140 | 0.007697 | 19.679063 | 0.264783 | 0.725340 |
| C_stress_delta | test | 7 | True | 0.041821 | 0.006031 | 4.907032 | 0.300628 | 0.701330 |
| C_stress_delta | validation | 123 | True | 0.048854 | 0.007540 | 19.512005 | 0.281770 | 0.705294 |
| C_stress_delta | test | 123 | True | 0.042514 | 0.005685 | 4.349988 | 0.311708 | 0.687085 |

## 4. Regime / bucket median

| 후보 | split | 구간 | FS median | severe median |
| --- | --- | --- | --- | --- |
| D_stock_fragility | validation | h1 | 0.324076 | 0.633914 |
| D_stock_fragility | validation | h2_h3 | 0.315675 | 0.665762 |
| D_stock_fragility | validation | h4_h5 | 0.303813 | 0.685090 |
| D_stock_fragility | validation | stress | 0.238938 | 0.651515 |
| D_stock_fragility | validation | vix_rising | 0.319915 | 0.674609 |
| D_stock_fragility | validation | breadth_worsening | 0.301803 | 0.697283 |
| D_stock_fragility | test | h1 | 0.374613 | 0.618683 |
| D_stock_fragility | test | h2_h3 | 0.360957 | 0.638632 |
| D_stock_fragility | test | h4_h5 | 0.354210 | 0.650270 |
| D_stock_fragility | test | stress | 0.279951 | 0.716437 |
| D_stock_fragility | test | vix_rising | 0.337509 | 0.661444 |
| D_stock_fragility | test | breadth_worsening | 0.350321 | 0.650736 |
| C_stress_delta | validation | h1 | 0.272696 | 0.677584 |
| C_stress_delta | validation | h2_h3 | 0.279829 | 0.697430 |
| C_stress_delta | validation | h4_h5 | 0.275778 | 0.712167 |
| C_stress_delta | validation | stress | 0.197640 | 0.712121 |
| C_stress_delta | validation | vix_rising | 0.288576 | 0.697674 |
| C_stress_delta | validation | breadth_worsening | 0.257897 | 0.736093 |
| C_stress_delta | test | h1 | 0.315922 | 0.677711 |
| C_stress_delta | test | h2_h3 | 0.309937 | 0.685925 |
| C_stress_delta | test | h4_h5 | 0.306492 | 0.694068 |
| C_stress_delta | test | stress | 0.226363 | 0.766625 |
| C_stress_delta | test | vix_rising | 0.280928 | 0.708886 |
| C_stress_delta | test | breadth_worsening | 0.304104 | 0.687034 |

## 5. 결론

- D_stock_fragility: `탈락 후보`
- C_stress_delta: `Stage 4-2 primary 후보`
- D 판단 근거: test false_safe/severe median이 Stage 4-2 하방 안정성 기준을 벗어나 seed 42 단일 신호 가능성이 크다.
- C 판단 근거: validation/test 양쪽에서 D보다 false_safe median이 낮고 severe median이 높아 seed 안정성이 더 좋다.
- D는 seed 42 단일 결과보다 validation/test median이 약해졌고 std도 커서 stock fragility 효과가 seed 안정적으로 유지됐다고 보기 어렵다.
- C는 원래 alpha 보존 비교 후보였지만, 이번 seed 재평가에서는 validation/test 하방 안정성도 D보다 안정적이었다.
- 따라서 다음 단계는 C_stress_delta를 Stage 4-2 primary 후보로 두고 seed 5개 또는 walk-forward로 확인하는 쪽이 자연스럽다.
- 별도 alpha-preserving challenger는 아직 확정하지 않는다. D는 validation fee/spread는 강하지만 test 하방 위험이 흔들려 alpha reference로도 조심스럽다.
- test에서 D의 false_safe/severe 붕괴가 보여 제품 후보 표현은 금지하고 feature overfit 의심으로 기록한다.
- ATR ablation 요청 상태: 본실험 우선으로 미실행

## 6. 산출물

- metrics: `docs\cp148_lm_1d_stage4_2_seed_stability_metrics.json`
- summary: `docs\cp148_lm_1d_stage4_2_seed_stability_summary.csv`
- logs/meta: `docs\cp148_lm_1d_stage4_2_seed_stability_logs`
- script: `ai/cp148_lm_1d_stage4_2_seed_stability.py`
