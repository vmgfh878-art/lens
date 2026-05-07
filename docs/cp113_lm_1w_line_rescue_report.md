# CP113-LM 1W line rescue feature set 비교

## 요약

- 상태: `PASS`
- CP112 기준 run_id: `patchtst-1W-584cc2586b4a`
- 최고 후보: `price_volatility_volume` / run_id `patchtst-1W-4b32afe5649c` / 판단 `return_direction_line_candidate`
- 범위: PatchTST 1W h4 line_model 전용. save-run, DB write, inference 저장, W&B, composite, 프론트 수정은 실행하지 않았다.

## 후보별 결과

| candidate | run_id | line_gate | class | ic | d_ic | spread | d_spread | false_safe | d_false_safe | severe_recall | d_recall | bias | dir_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_features_baseline | patchtst-1W-d9ef659b9c8d | True | risk_conservative_line_candidate | -0.009222 | 0.005383 | -0.004590 | 0.002464 | 0.222509 | -0.016753 | 0.765612 | 0.005272 | -0.103006 | 0.452749 |
| no_fundamentals | patchtst-1W-5c5af071918d | True | return_direction_line_candidate | 0.019180 | 0.033785 | 0.006797 | 0.013851 | 0.223797 | -0.015464 | 0.782238 | 0.021898 | -0.111378 | 0.455326 |
| price_volatility_volume | patchtst-1W-4b32afe5649c | True | return_direction_line_candidate | 0.014409 | 0.029014 | 0.009531 | 0.016585 | 0.188574 | -0.050687 | 0.809002 | 0.048662 | -0.113134 | 0.455670 |

## CP112 smoke 대비 개선/악화

### full_features_baseline

| metric | CP112 | current | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | -0.014605 | -0.009222 | 0.005383 | improved |
| ic_ir | -0.071242 | -0.043459 | 0.027782 | improved |
| ic_t_stat | -0.551837 | -0.336636 | 0.215201 | improved |
| long_short_spread | -0.007054 | -0.004590 | 0.002464 | improved |
| spread_ir | -0.120777 | -0.080589 | 0.040188 | improved |
| spread_t_stat | -0.935536 | -0.624240 | 0.311296 | improved |
| direction_accuracy | 0.450515 | 0.452749 | 0.002234 | improved |
| false_safe_tail_rate | 0.239261 | 0.222509 | -0.016753 | improved |
| false_safe_severe_rate | 0.239659 | 0.234388 | -0.005272 | improved |
| severe_downside_recall | 0.760341 | 0.765612 | 0.005272 | improved |
| downside_capture_rate | 0.224227 | 0.219287 | -0.004940 | worsened |
| conservative_bias | -0.124406 | -0.103006 | 0.021400 | less_conservative |
| upside_sacrifice | 0.231413 | 0.204266 | -0.027147 | improved |
| mae | 0.171447 | 0.141040 | -0.030407 | improved |
| smape | 1.606140 | 1.575310 | -0.030831 | improved |
| fee_adjusted_return | -0.484434 | -0.381137 | 0.103297 | improved |
| fee_adjusted_sharpe | -0.160782 | -0.112870 | 0.047912 | improved |

### no_fundamentals

| metric | CP112 | current | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | -0.014605 | 0.019180 | 0.033785 | improved |
| ic_ir | -0.071242 | 0.113982 | 0.185223 | improved |
| ic_t_stat | -0.551837 | 0.882897 | 1.434734 | improved |
| long_short_spread | -0.007054 | 0.006797 | 0.013851 | improved |
| spread_ir | -0.120777 | 0.142532 | 0.263310 | improved |
| spread_t_stat | -0.935536 | 1.104052 | 2.039588 | improved |
| direction_accuracy | 0.450515 | 0.455326 | 0.004811 | improved |
| false_safe_tail_rate | 0.239261 | 0.223797 | -0.015464 | improved |
| false_safe_severe_rate | 0.239659 | 0.217762 | -0.021898 | improved |
| severe_downside_recall | 0.760341 | 0.782238 | 0.021898 | improved |
| downside_capture_rate | 0.224227 | 0.227663 | 0.003436 | improved |
| conservative_bias | -0.124406 | -0.111378 | 0.013028 | less_conservative |
| upside_sacrifice | 0.231413 | 0.212930 | -0.018483 | improved |
| mae | 0.171447 | 0.152988 | -0.018459 | improved |
| smape | 1.606140 | 1.579444 | -0.026696 | improved |
| fee_adjusted_return | -0.484434 | 0.246048 | 0.730481 | improved |
| fee_adjusted_sharpe | -0.160782 | 0.101043 | 0.261825 | improved |

### price_volatility_volume

| metric | CP112 | current | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | -0.014605 | 0.014409 | 0.029014 | improved |
| ic_ir | -0.071242 | 0.086992 | 0.158234 | improved |
| ic_t_stat | -0.551837 | 0.673840 | 1.225677 | improved |
| long_short_spread | -0.007054 | 0.009531 | 0.016585 | improved |
| spread_ir | -0.120777 | 0.167036 | 0.287813 | improved |
| spread_t_stat | -0.935536 | 1.293856 | 2.229393 | improved |
| direction_accuracy | 0.450515 | 0.455670 | 0.005155 | improved |
| false_safe_tail_rate | 0.239261 | 0.188574 | -0.050687 | improved |
| false_safe_severe_rate | 0.239659 | 0.190998 | -0.048662 | improved |
| severe_downside_recall | 0.760341 | 0.809002 | 0.048662 | improved |
| downside_capture_rate | 0.224227 | 0.224656 | 0.000430 | improved |
| conservative_bias | -0.124406 | -0.113134 | 0.011272 | less_conservative |
| upside_sacrifice | 0.231413 | 0.209404 | -0.022009 | improved |
| mae | 0.171447 | 0.143783 | -0.027665 | improved |
| smape | 1.606140 | 1.575837 | -0.030303 | improved |
| fee_adjusted_return | -0.484434 | 0.439559 | 0.923993 | improved |
| fee_adjusted_sharpe | -0.160782 | 0.135547 | 0.296330 | improved |

## h1_h4 bucket

| candidate | ic_mean | long_short_spread | false_safe_tail_rate | severe_downside_recall |
| --- | --- | --- | --- | --- |
| full_features_baseline | -0.010334 | -0.005187 | 0.178333 | 0.809981 |
| no_fundamentals | 0.024346 | 0.008105 | 0.157500 | 0.833013 |
| price_volatility_volume | 0.012176 | 0.001567 | 0.153333 | 0.842610 |

## 제품/다음 CP 판단

- 1W 수익 방향선 가능 여부: `True`
- 1W 중기 위험/보수선 가능 여부: `True`
- 제품 후보 저장 진행: `not_yet`
- 저장 판단 사유: 수익 방향선 후보는 보이나 CP113은 save-run 금지 smoke 비교이므로 저장 전용 CP에서 재현 후 저장한다.
- 다음 CP 추천: 가장 좋은 1W 수익 방향선 후보를 같은 설정으로 epochs 5, save-run 전용 후보 CP에서 재현한다.
