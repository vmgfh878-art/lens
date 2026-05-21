# CP114-LM 1W line 후보 확장

## 요약

- 상태: `PASS`
- CP113 기준 후보: `price_volatility_volume` / `patchtst-1W-4b32afe5649c`
- best h4 후보: `h4_pvv_patch16_stride8` / `patchtst-1W-b513e826408e` / `product_line_candidate`
- 범위: PatchTST 1W line_model 전용. save-run, DB write, inference 저장, W&B, composite, 프론트 수정은 실행하지 않았다.

## 후보 결과

| candidate | run_id | gate | class | ic | d_ic | spread | d_spread | false_safe | d_false_safe | severe | d_severe | bias | upside | fee_ret | fee_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h4_pvv_patch16_stride8 | patchtst-1W-b513e826408e | True | product_line_candidate | 0.011420 | -0.002988 | 0.008156 | -0.001375 | 0.166667 | -0.021907 | 0.828062 | 0.019059 | -0.115884 | 0.217676 | 0.320460 | 0.107028 |
| h4_no_fundamentals_patch16_stride8 | patchtst-1W-12891738db0d | True | phase1_watch | 0.025148 | 0.010740 | 0.013697 | 0.004166 | 0.225086 | 0.036512 | 0.773723 | -0.035280 | -0.112085 | 0.209100 | 0.856072 | 0.234635 |
| h4_pvv_patch32_stride16 | patchtst-1W-d73cdfca969c | True | phase1_watch | -0.010150 | -0.024558 | 0.004095 | -0.005436 | 0.289089 | 0.100515 | 0.698702 | -0.110300 | -0.066193 | 0.164311 | 0.054038 | 0.044991 |
| h8_pvv_patch16_stride8 | patchtst-1W-57a3578c4cd1 | False | h8_feasibility_fail | 0.023124 | 0.008715 | -0.011239 | -0.020770 | 0.221972 | 0.033398 | 0.771315 | -0.037687 | -0.099368 | 0.234459 | -0.581434 | -0.181481 |

## CP113 best 대비 개선/악화

### h4_pvv_patch16_stride8

| metric | CP113 best | current | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.014409 | 0.011420 | -0.002988 | worsened |
| ic_ir | 0.086992 | 0.066035 | -0.020957 | worsened |
| ic_t_stat | 0.673840 | 0.511506 | -0.162334 | worsened |
| long_short_spread | 0.009531 | 0.008156 | -0.001375 | worsened |
| spread_ir | 0.167036 | 0.136633 | -0.030403 | worsened |
| spread_t_stat | 1.293856 | 1.058352 | -0.235504 | worsened |
| direction_accuracy | 0.455670 | 0.456357 | 0.000687 | improved |
| false_safe_tail_rate | 0.188574 | 0.166667 | -0.021907 | improved |
| false_safe_severe_rate | 0.190998 | 0.171938 | -0.019059 | improved |
| severe_downside_recall | 0.809002 | 0.828062 | 0.019059 | improved |
| downside_capture_rate | 0.224656 | 0.227019 | 0.002363 | improved |
| conservative_bias | -0.113134 | -0.115884 | -0.002750 | more_conservative |
| upside_sacrifice | 0.209404 | 0.217676 | 0.008273 | worsened |
| mae | 0.143783 | 0.142467 | -0.001316 | improved |
| smape | 1.575837 | 1.581273 | 0.005436 | worsened |
| fee_adjusted_return | 0.439559 | 0.320460 | -0.119099 | worsened |
| fee_adjusted_sharpe | 0.135547 | 0.107028 | -0.028519 | worsened |

### h4_no_fundamentals_patch16_stride8

| metric | CP113 best | current | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.014409 | 0.025148 | 0.010740 | improved |
| ic_ir | 0.086992 | 0.145833 | 0.058841 | improved |
| ic_t_stat | 0.673840 | 1.129616 | 0.455777 | improved |
| long_short_spread | 0.009531 | 0.013697 | 0.004166 | improved |
| spread_ir | 0.167036 | 0.276078 | 0.109042 | improved |
| spread_t_stat | 1.293856 | 2.138490 | 0.844634 | improved |
| direction_accuracy | 0.455670 | 0.466495 | 0.010825 | improved |
| false_safe_tail_rate | 0.188574 | 0.225086 | 0.036512 | worsened |
| false_safe_severe_rate | 0.190998 | 0.226277 | 0.035280 | worsened |
| severe_downside_recall | 0.809002 | 0.773723 | -0.035280 | worsened |
| downside_capture_rate | 0.224656 | 0.228093 | 0.003436 | improved |
| conservative_bias | -0.113134 | -0.112085 | 0.001049 | less_conservative |
| upside_sacrifice | 0.209404 | 0.209100 | -0.000303 | improved |
| mae | 0.143783 | 0.152662 | 0.008879 | worsened |
| smape | 1.575837 | 1.579399 | 0.003561 | worsened |
| fee_adjusted_return | 0.439559 | 0.856072 | 0.416513 | improved |
| fee_adjusted_sharpe | 0.135547 | 0.234635 | 0.099088 | improved |

### h4_pvv_patch32_stride16

| metric | CP113 best | current | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.014409 | -0.010150 | -0.024558 | worsened |
| ic_ir | 0.086992 | -0.044186 | -0.131178 | worsened |
| ic_t_stat | 0.673840 | -0.342262 | -1.016101 | worsened |
| long_short_spread | 0.009531 | 0.004095 | -0.005436 | worsened |
| spread_ir | 0.167036 | 0.061878 | -0.105159 | worsened |
| spread_t_stat | 1.293856 | 0.479301 | -0.814555 | worsened |
| direction_accuracy | 0.455670 | 0.465120 | 0.009450 | improved |
| false_safe_tail_rate | 0.188574 | 0.289089 | 0.100515 | worsened |
| false_safe_severe_rate | 0.190998 | 0.301298 | 0.110300 | worsened |
| severe_downside_recall | 0.809002 | 0.698702 | -0.110300 | worsened |
| downside_capture_rate | 0.224656 | 0.220361 | -0.004296 | worsened |
| conservative_bias | -0.113134 | -0.066193 | 0.046941 | less_conservative |
| upside_sacrifice | 0.209404 | 0.164311 | -0.045093 | improved |
| mae | 0.143783 | 0.110652 | -0.033131 | improved |
| smape | 1.575837 | 1.541831 | -0.034006 | improved |
| fee_adjusted_return | 0.439559 | 0.054038 | -0.385521 | worsened |
| fee_adjusted_sharpe | 0.135547 | 0.044991 | -0.090556 | worsened |

### h8_pvv_patch16_stride8

| metric | CP113 best | current | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.014409 | 0.023124 | 0.008715 | improved |
| ic_ir | 0.086992 | 0.153643 | 0.066651 | improved |
| ic_t_stat | 0.673840 | 1.190114 | 0.516274 | improved |
| long_short_spread | 0.009531 | -0.011239 | -0.020770 | worsened |
| spread_ir | 0.167036 | -0.166075 | -0.333111 | worsened |
| spread_t_stat | 1.293856 | -1.286413 | -2.580270 | worsened |
| direction_accuracy | 0.455670 | 0.457732 | 0.002062 | improved |
| false_safe_tail_rate | 0.188574 | 0.221972 | 0.033398 | worsened |
| false_safe_severe_rate | 0.190998 | 0.228685 | 0.037687 | worsened |
| severe_downside_recall | 0.809002 | 0.771315 | -0.037687 | worsened |
| downside_capture_rate | 0.224656 | 0.226052 | 0.001396 | improved |
| conservative_bias | -0.113134 | -0.099368 | 0.013767 | less_conservative |
| upside_sacrifice | 0.209404 | 0.234459 | 0.025055 | worsened |
| mae | 0.143783 | 0.137466 | -0.006317 | improved |
| smape | 1.575837 | 1.537032 | -0.038806 | improved |
| fee_adjusted_return | 0.439559 | -0.581434 | -1.020993 | worsened |
| fee_adjusted_sharpe | 0.135547 | -0.181481 | -0.317029 | worsened |

## horizon bucket

| candidate | bucket | ic_mean | long_short_spread | false_safe_tail_rate | severe_downside_recall |
| --- | --- | --- | --- | --- | --- |
| h4_pvv_patch16_stride8 | h1_h5 | 0.004270 | -0.003638 | 0.130000 | 0.857965 |
| h4_no_fundamentals_patch16_stride8 | h1_h5 | 0.024892 | 0.007637 | 0.155000 | 0.838772 |
| h4_pvv_patch32_stride16 | h1_h5 | -0.005444 | 0.001232 | 0.238333 | 0.727447 |
| h8_pvv_patch16_stride8 | h1_h5 | 0.009264 | 0.002119 | 0.149167 | 0.844828 |
| h8_pvv_patch16_stride8 | h6_h10 | 0.023759 | -0.006809 | 0.187500 | 0.784394 |

## 판정

- 제품 line 후보 존재: `True`
- risk-only 후보 존재: `False`
- h8 feasibility/watch 기록: `True`
- 제품 후보 저장 진행: `yes_next_cp`
- 사유: h4 후보가 제품 line 기준을 통과했다. CP114는 save-run 금지이므로 다음 CP에서 같은 설정으로 저장 후보 재현이 필요하다.
- 다음 CP 추천: h4_pvv_patch16_stride8를 1W h4 line 저장 전용 CP로 재현한다. 같은 yfinance snapshot/hash, epochs 5, save-run true 여부는 별도 지시가 필요하다.
