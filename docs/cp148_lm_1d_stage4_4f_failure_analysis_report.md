# CP148-LM-1D Stage 4-4F failure analysis 보고서

- 작성 시각: 2026-05-10T20:38:37
- 범위: Stage 4-4 trial006/trial024의 기존 6개 checkpoint test split 진단
- 금지 작업 준수: 새 학습, Optuna, product save-run, DB write, inference 저장, live fetch, band/composite 모두 미실행

## 1. 한 줄 결론

seed 42 붕괴는 단순 selector/bias 문제가 아니라, non-stress/quiet tail에서 실제 하락을 강한 양수 score로 본 feature blind 문제가 1차 원인이다.

## 2. seed 42 실패 한 문장

seed 42는 다른 seed보다 덜 보수적으로 떴지만, false-safe 대부분이 0 근처가 아니라 0.005 이상 강한 양수 score였기 때문에 severe recall이 같이 무너졌다.

## 3. 판정

- 1차 판정: `feature_blind_problem`
- 부가 태그: `feature_blind_problem, quiet_idiosyncratic_tail`
- seed42 덜 보수적 여부: `True`
- seed42 false_safe 0~0.005 근처 평균 비중: `0.099355`
- seed42 false_safe 0.005 이상 강한 양수 평균 비중: `0.900645`
- seed42 non-stress minus stress false_safe gap: `0.082085`
- seed42 h1 minus h4_h5 false_safe gap: `0.001039`

## 4. 지표와 margin 요약

| 후보 | seed | IC | spread | false_safe | severe | bias | 0~0.001 | 0.001~0.003 | 0.003~0.005 | 0.005+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | 7 | 0.034465 | 0.004774 | 0.268859 | 0.732031 | -0.031862 | 0.023182 | 0.046861 | 0.042944 | 0.887013 |
| trial006_c_balanced | 42 | 0.044837 | 0.006704 | 0.352043 | 0.651559 | -0.017570 | 0.020581 | 0.041161 | 0.040202 | 0.898056 |
| trial006_c_balanced | 123 | 0.042207 | 0.005298 | 0.281680 | 0.717994 | -0.029967 | 0.022189 | 0.045638 | 0.043262 | 0.888912 |
| trial024_c_risk | 7 | 0.044147 | 0.005861 | 0.289891 | 0.712520 | -0.027723 | 0.023648 | 0.045248 | 0.044445 | 0.886658 |
| trial024_c_risk | 42 | 0.041446 | 0.006389 | 0.374070 | 0.626448 | -0.014486 | 0.018933 | 0.038582 | 0.039251 | 0.903234 |
| trial024_c_risk | 123 | 0.042502 | 0.005812 | 0.295006 | 0.706533 | -0.026421 | 0.023337 | 0.043773 | 0.042787 | 0.890103 |

## 5. stress / non-stress 분해

| 후보 | seed | 구간 | tail_count | false_safe | severe_count | severe_recall |
| --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | 7 | stress | 16402 | 0.193513 | 5353 | 0.790024 |
| trial006_c_balanced | 7 | vix_rising | 89345 | 0.251967 | 28604 | 0.746749 |
| trial006_c_balanced | 7 | breadth_worsening | 99749 | 0.273807 | 30879 | 0.723307 |
| trial006_c_balanced | 7 | non_stress | 155435 | 0.276810 | 43444 | 0.722033 |
| trial006_c_balanced | 7 | non_vix_rising | 82492 | 0.287155 | 20193 | 0.705046 |
| trial006_c_balanced | 7 | non_breadth_worsening | 72088 | 0.262013 | 17918 | 0.740150 |
| trial006_c_balanced | 42 | stress | 16402 | 0.274845 | 5353 | 0.716981 |
| trial006_c_balanced | 42 | vix_rising | 89345 | 0.340836 | 28604 | 0.663858 |
| trial006_c_balanced | 42 | breadth_worsening | 99749 | 0.350169 | 30879 | 0.648402 |
| trial006_c_balanced | 42 | non_stress | 155435 | 0.360189 | 43444 | 0.646948 |
| trial006_c_balanced | 42 | non_vix_rising | 82492 | 0.364181 | 20193 | 0.641559 |
| trial006_c_balanced | 42 | non_breadth_worsening | 72088 | 0.354636 | 17918 | 0.665364 |
| trial006_c_balanced | 123 | stress | 16402 | 0.208328 | 5353 | 0.774145 |
| trial006_c_balanced | 123 | vix_rising | 89345 | 0.265734 | 28604 | 0.731856 |
| trial006_c_balanced | 123 | breadth_worsening | 99749 | 0.280674 | 30879 | 0.710936 |
| trial006_c_balanced | 123 | non_stress | 155435 | 0.289420 | 43444 | 0.707946 |
| trial006_c_balanced | 123 | non_vix_rising | 82492 | 0.298950 | 20193 | 0.691626 |
| trial006_c_balanced | 123 | non_breadth_worsening | 72088 | 0.283071 | 17918 | 0.722569 |
| trial024_c_risk | 7 | stress | 16402 | 0.223326 | 5353 | 0.756959 |
| trial024_c_risk | 7 | vix_rising | 89345 | 0.270569 | 28604 | 0.730422 |
| trial024_c_risk | 7 | breadth_worsening | 99749 | 0.293246 | 30879 | 0.704071 |
| trial024_c_risk | 7 | non_stress | 155435 | 0.296915 | 43444 | 0.708659 |
| trial024_c_risk | 7 | non_vix_rising | 82492 | 0.310818 | 20193 | 0.690635 |
| trial024_c_risk | 7 | non_breadth_worsening | 72088 | 0.285249 | 17918 | 0.730997 |
| trial024_c_risk | 42 | stress | 16402 | 0.302768 | 5353 | 0.681674 |
| trial024_c_risk | 42 | vix_rising | 89345 | 0.358005 | 28604 | 0.645015 |
| trial024_c_risk | 42 | breadth_worsening | 99749 | 0.378931 | 30879 | 0.620163 |
| trial024_c_risk | 42 | non_stress | 155435 | 0.381594 | 43444 | 0.620247 |
| trial024_c_risk | 42 | non_vix_rising | 82492 | 0.391468 | 20193 | 0.601446 |
| trial024_c_risk | 42 | non_breadth_worsening | 72088 | 0.367343 | 17918 | 0.638743 |
| trial024_c_risk | 123 | stress | 16402 | 0.226558 | 5353 | 0.760882 |
| trial024_c_risk | 123 | vix_rising | 89345 | 0.278035 | 28604 | 0.722556 |
| trial024_c_risk | 123 | breadth_worsening | 99749 | 0.297898 | 30879 | 0.701415 |
| trial024_c_risk | 123 | non_stress | 155435 | 0.302229 | 43444 | 0.700741 |
| trial024_c_risk | 123 | non_vix_rising | 82492 | 0.313388 | 20193 | 0.685782 |
| trial024_c_risk | 123 | non_breadth_worsening | 72088 | 0.291005 | 17918 | 0.717547 |

## 6. horizon 분해

| 후보 | seed | bucket | tail_count | false_safe | severe_count | severe_recall |
| --- | --- | --- | --- | --- | --- | --- |
| trial006_c_balanced | 7 | h1 | 18114 | 0.276250 | 2864 | 0.701466 |
| trial006_c_balanced | 7 | h2_h3 | 66529 | 0.275414 | 16478 | 0.715682 |
| trial006_c_balanced | 7 | h4_h5 | 87194 | 0.262323 | 29455 | 0.739942 |
| trial006_c_balanced | 42 | h1 | 18114 | 0.357403 | 2864 | 0.640014 |
| trial006_c_balanced | 42 | h2_h3 | 66529 | 0.354327 | 16478 | 0.649351 |
| trial006_c_balanced | 42 | h4_h5 | 87194 | 0.349187 | 29455 | 0.659005 |
| trial006_c_balanced | 123 | h1 | 18114 | 0.285194 | 2864 | 0.694134 |
| trial006_c_balanced | 123 | h2_h3 | 66529 | 0.275684 | 16478 | 0.714953 |
| trial006_c_balanced | 123 | h4_h5 | 87194 | 0.285524 | 29455 | 0.717399 |
| trial024_c_risk | 7 | h1 | 18114 | 0.291984 | 2864 | 0.686103 |
| trial024_c_risk | 7 | h2_h3 | 66529 | 0.282163 | 16478 | 0.716895 |
| trial024_c_risk | 7 | h4_h5 | 87194 | 0.295353 | 29455 | 0.715023 |
| trial024_c_risk | 42 | h1 | 18114 | 0.370045 | 2864 | 0.615922 |
| trial024_c_risk | 42 | h2_h3 | 66529 | 0.372394 | 16478 | 0.625501 |
| trial024_c_risk | 42 | h4_h5 | 87194 | 0.376184 | 29455 | 0.628892 |
| trial024_c_risk | 123 | h1 | 18114 | 0.299713 | 2864 | 0.687849 |
| trial024_c_risk | 123 | h2_h3 | 66529 | 0.289363 | 16478 | 0.708703 |
| trial024_c_risk | 123 | h4_h5 | 87194 | 0.298335 | 29455 | 0.708471 |

## 7. feature blind 확인

- feature 비교는 test split의 마지막 입력 시점 값을 사용했다.
- 값의 스케일은 모델 입력과 같은 train-normalized scale이다.
- false_safe missed severe와 correctly caught severe의 feature 평균/분위수 차이는 metrics JSON의 `feature_blind_analysis`에 기록했다.

## 8. 날짜/종목 집중도

- top false_safe dates: `docs\cp148_lm_1d_stage4_4f_false_safe_top_dates.csv`
- top false_safe tickers: `docs\cp148_lm_1d_stage4_4f_false_safe_top_tickers.csv`
- sector/industry 사용 여부: `{'sector_source': None, 'sector_columns': []}`

## 9. 다음 실험 1개

- 추천: Stage 4-5 단일 실험: C + overextension/quiet-tail fragility pack
- 내용: C_stress_delta는 유지하되, non-stress에서 좋아 보이는 종목이 갑자기 tail로 빠지는 패턴을 잡기 위해 runup/overextension 계열 파생 피처를 추가한 3-seed 단일 후보를 검증한다. 후보 피처는 20일 runup, ma_20/ma_60 양의 괴리, RSI overbought, BB upper excess, 최근 max drawdown 또는 max down day 중 local parquet에서 계산 가능한 값으로 제한한다.
- seed 안정성 통과 가능성을 높이는 이유: seed 42 false-safe의 약 90%가 0.005 이상 강한 양수 score였고, missed severe는 caught severe보다 ma_60_ratio, ma_20_ratio, RSI, bb_position, log_return이 높았다. 즉 모델이 조용한 상승/과열 종목의 급락을 안전 신호로 읽었으므로, overextension을 명시 피처로 주면 seed별 bias drift보다 구조적인 blind spot을 줄일 가능성이 높다.

## 10. 산출물

- metrics: `docs\cp148_lm_1d_stage4_4f_failure_analysis_metrics.json`
- top dates: `docs\cp148_lm_1d_stage4_4f_false_safe_top_dates.csv`
- top tickers: `docs\cp148_lm_1d_stage4_4f_false_safe_top_tickers.csv`
- script: `ai/cp148_lm_1d_stage4_4f_failure_analysis.py`
