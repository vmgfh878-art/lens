# CP202.1 Band Baseline Comparison

이번 비교는 새 학습 없이 기존 1D/1W band audit 집계와 local yfinance parquet 기반 통계 baseline을 같은 표 형식으로 정리한 reference 비교다.
1W도 사용자 추가 지시에 따라 1D와 같은 형식으로 포함했다.

- 최종 라벨: `WARN_CP202_1_BAND_MARGINAL_OVER_BASELINE`
- price parquet: `C:\Users\user\lens\data\parquet\price_data_yfinance_500.parquet`
- GARCH 구현: `arch 패키지 미사용. alpha=0.05, beta=0.90 고정 closed-form GARCH(1,1) fallback.`
- torch import가 현재 세션에서 실패했기 때문에 기존 checkpoint forward 재실행은 하지 않았다.
- 우리 band의 A/B/regime/lead-lag 수치는 기존 CP202 aggregate를 read-only로 재사용했다.
- 통계 baseline은 같은 yfinance local parquet, 같은 fold test 구간, 같은 q 설정으로 closed-form 계산했다.
- 우리 band의 월별 coverage stability는 CP202 row-level frame이 저장되어 있지 않아 산출 불가로 비워 두었다.

## Quantile And k

| timeframe | target | q_low | q_high | nominal_coverage | normal k | calibration |
|---|---|---:|---:|---:|---:|---|
| 1D | tide_s60_q15_param | 0.15 | 0.85 | 0.700 | 1.036433 | lower_focused |
| 1W | tide_s60_q10_q90_param | 0.10 | 0.90 | 0.800 | 1.281552 | walk_forward_lower_calibration |

## Results

### 1D

| baseline | source | samples | cov_abs | lower | upper | stress_cov_abs | vix_gt30_cov_abs | vix_gt30_lower | future_auc | width_abs_ic | past_vol_ic | p90_width | month_worst |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OUR_TIDE_QUANTILE | existing_cp202_aggregate_read_only | 2655465 | 0.001034 | 0.141943 | 0.157024 | 0.037651 | 0.103206 | 0.108137 | 0.518364 | 0.249385 | 0.702252 | 0.092930 |  |
| B1_BOLLINGER_RETURN | local_parquet_closed_form_baseline | 934115 | 0.053441 | 0.177341 | 0.176100 | 0.074166 | 0.110189 | 0.098884 | 0.511596 | 0.208099 | 0.834934 | 0.109835 | 0.125010 |
| B2_HISTORICAL_QUANTILE_ROLLING | local_parquet_closed_form_baseline | 934115 | 0.049069 | 0.177793 | 0.171277 | 0.111841 | 0.194848 | 0.150487 | 0.496939 | 0.240607 | 0.665717 | 0.096037 | 0.155196 |
| B3_GARCH11_CLOSED_FORM | local_parquet_closed_form_baseline | 934115 | 0.039883 | 0.134012 | 0.126104 | 0.005143 | 0.050973 | 0.113881 | 0.519415 | 0.257948 | 0.815050 | 0.111518 | 0.120823 |

### 1W

| baseline | source | samples | cov_abs | lower | upper | stress_cov_abs | vix_gt30_cov_abs | vix_gt30_lower | future_auc | width_abs_ic | past_vol_ic | p90_width | month_worst |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OUR_TIDE_QUANTILE | existing_cp202_aggregate_read_only | 411312 | 0.034316 | 0.117966 | 0.116350 | 0.072094 | 0.097364 | 0.036277 | 0.526238 | 0.236884 | 0.617472 | 0.210313 |  |
| B1_BOLLINGER_RETURN | local_parquet_closed_form_baseline | 150600 | 0.077722 | 0.145193 | 0.132530 | 0.141420 | 0.178447 | 0.037996 | 0.518166 | 0.231337 | 0.860786 | 0.258523 | 0.207828 |
| B2_HISTORICAL_QUANTILE_ROLLING | local_parquet_closed_form_baseline | 150600 | 0.071534 | 0.144376 | 0.127158 | 0.144875 | 0.172226 | 0.041695 | 0.489455 | 0.275443 | 0.771789 | 0.239997 | 0.217720 |
| B3_GARCH11_CLOSED_FORM | local_parquet_closed_form_baseline | 150600 | 0.025471 | 0.122902 | 0.102570 | 0.061086 | 0.068325 | 0.030935 | 0.487531 | 0.284764 | 0.850810 | 0.256022 | 0.159454 |

## 차원별 해석

### Stress regime

| timeframe | ours stress cov_abs | best baseline stress cov_abs | ours vs BB stress cov_abs | 해석 |
|---|---:|---:|---:|---|
| 1D | 0.037651 | 0.005143 | -0.036515 | baseline과 비슷하거나 열위 |
| 1W | 0.072094 | 0.061086 | -0.069326 | baseline과 비슷하거나 열위 |

### Stop-loss 방향성

| timeframe | ours VIX>30 lower | best baseline VIX>30 lower | ours lower-upper delta | 해석 |
|---|---:|---:|---:|---|
| 1D | 0.108137 | 0.098884 | -0.015081 | 하방 보수성 우위는 제한적 |
| 1W | 0.036277 | 0.030935 | 0.001617 | 하방 보수성 우위는 제한적 |

### Width future severe AUC

| timeframe | ours AUC | best baseline AUC | ours past-vol IC | 해석 |
|---|---:|---:|---:|---|
| 1D | 0.518364 | 0.519415 | 0.702252 | 선행성은 약하고 범위 신호로 해석하는 편이 맞음 |
| 1W | 0.526238 | 0.518166 | 0.617472 | 선행성은 약하고 범위 신호로 해석하는 편이 맞음 |

## 한 페이지 결론

- 이번 작업의 질문은 우리 band가 무엇과 비교해서 의미 있는지다.
- 결과가 CONFIRM이면 CP202 WARN은 표준 baseline보다 나은 정상 작동 band를 더 조심스럽게 설명해야 한다는 뜻이다.
- 결과가 WARN이면 deep band가 모든 차원에서 통계 baseline을 명확히 압도하지는 못한다는 뜻이다.
- 이 경우 product 역할은 단일 forecast가 아니라 calibrated risk interval 및 범위 신호로 제한해 설명하는 편이 맞다.
- 1W는 같은 형식으로 비교했지만 product 승격 판단은 CP178-WFLOCK의 strict/대칭 기준 선택과 별도로 사용자 판단이 필요하다.

## 산출물

- CSV: `C:\Users\user\lens\docs\cp202_1_band_baseline_comparison.csv`
- metrics JSON: `C:\Users\user\lens\docs\cp202_1_band_baseline_comparison_metrics.json`
- progress: `C:\Users\user\lens\docs\cp202_1_progress.log`
