# CP136-LM 1W Line Horizon / Feature Expansion 재실험

## 요약

- 상태: `PASS`
- recommended_default: `h4_pvv_patch16_stride8_repro` / run_id `patchtst-1W-c4c34d4f0869`
- 후보 선택은 validation line_metrics 기준이며 test는 직접 선택에 쓰지 않았다.
- save-run, DB write, inference 저장, W&B, composite, 프론트 수정, live yfinance fetch, EODHD 호출은 하지 않았다.

## 데이터 기준

- source_data_hash: `13a7f83d`
- CP113 대비 source hash 변경: `True`
- context checksum: `ecb532122fca5eee`
- indicator snapshot mtime: `2026-05-06T15:14:01`
- feature/target NaN/Inf: `0` / `0`
- context_light feature_set: `False`

## 후보 결과

| candidate | class | run_id | model | h | feature_set | gate | val_ic | d_ic | val_spread | val_false_safe | val_severe | val_fee |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h4_pvv_patch16_stride8_repro | recommended_default | patchtst-1W-c4c34d4f0869 | patchtst | 4 | price_volatility_volume | True | 0.029400 | 0.017286 | 0.005852 | 0.079057 | 0.901293 | 0.183052 |
| h4_no_fundamentals_patch16_stride8 | selectable_verified | patchtst-1W-2da47c1beb62 | patchtst | 4 | no_fundamentals | True | 0.027627 | 0.015513 | 0.010193 | 0.128631 | 0.858407 | 0.525498 |
| h4_pvv_dense_stride4 | selectable_verified | patchtst-1W-f5e01d29d16f | patchtst | 4 | price_volatility_volume | True | 0.008846 | -0.003268 | 0.005044 | 0.077528 | 0.914227 | 0.083525 |
| h4_tide_pvv | rejected | tide-1W-95f68d565a56 | tide | 4 | price_volatility_volume | True | 0.047596 | 0.035482 | 0.013594 | 0.310330 | 0.687543 | 0.797689 |
| h6_pvv_patch16_stride8 | rejected | patchtst-1W-d5135afd47c8 | patchtst | 6 | price_volatility_volume | False | -0.044170 | -0.056284 | -0.008997 | 0.101340 | 0.894182 | -0.533175 |
| h8_pvv_patch16_stride8 | experiment_record | patchtst-1W-3112176ae412 | patchtst | 8 | price_volatility_volume | False | -0.040308 | -0.052422 | -0.022890 | 0.114011 | 0.866510 | -0.794189 |
| h8_no_fundamentals_patch16_stride8 | experiment_record | patchtst-1W-876e2dfcf633 | patchtst | 8 | no_fundamentals | False | 0.019445 | 0.007331 | -0.004881 | 0.119690 | 0.879287 | -0.409349 |
| h4_context_light_patch16_stride8 | design_needed | None | patchtst | 4 | context_light | None |  |  |  |  |  |  |

## CP113 best 대비 validation 개선/악화

### h4_pvv_patch16_stride8_repro

| metric | CP113 best val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.012114 | 0.029400 | 0.017286 | improved |
| ic_ir | 0.069818 | 0.158777 | 0.088958 | improved |
| ic_t_stat | 0.536284 | 1.219586 | 0.683302 | improved |
| long_short_spread | 0.000676 | 0.005852 | 0.005176 | improved |
| spread_ir | 0.014745 | 0.133667 | 0.118923 | improved |
| spread_t_stat | 0.113256 | 1.026718 | 0.913462 | improved |
| fee_adjusted_return | -0.133774 | 0.183052 | 0.316826 | improved |
| fee_adjusted_sharpe | -0.030813 | 0.087691 | 0.118504 | improved |
| false_safe_tail_rate | 0.095217 | 0.079057 | -0.016161 | improved |
| false_safe_severe_rate | 0.116406 | 0.098707 | -0.017699 | improved |
| severe_downside_recall | 0.883594 | 0.901293 | 0.017699 | improved |
| conservative_bias | -0.167578 | -0.171929 | -0.004351 | more_conservative |
| upside_sacrifice | 0.256570 | 0.262746 | 0.006177 | worsened |

### h4_no_fundamentals_patch16_stride8

| metric | CP113 best val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.012114 | 0.027627 | 0.015513 | improved |
| ic_ir | 0.069818 | 0.178558 | 0.108740 | improved |
| ic_t_stat | 0.536284 | 1.371533 | 0.835250 | improved |
| long_short_spread | 0.000676 | 0.010193 | 0.009517 | improved |
| spread_ir | 0.014745 | 0.240599 | 0.225854 | improved |
| spread_t_stat | 0.113256 | 1.848073 | 1.734817 | improved |
| fee_adjusted_return | -0.133774 | 0.525498 | 0.659272 | improved |
| fee_adjusted_sharpe | -0.030813 | 0.192155 | 0.222967 | improved |
| false_safe_tail_rate | 0.095217 | 0.128631 | 0.033413 | worsened |
| false_safe_severe_rate | 0.116406 | 0.141593 | 0.025187 | worsened |
| severe_downside_recall | 0.883594 | 0.858407 | -0.025187 | worsened |
| conservative_bias | -0.167578 | -0.176360 | -0.008782 | more_conservative |
| upside_sacrifice | 0.256570 | 0.263474 | 0.006904 | worsened |

### h4_pvv_dense_stride4

| metric | CP113 best val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.012114 | 0.008846 | -0.003268 | worsened |
| ic_ir | 0.069818 | 0.048517 | -0.021302 | worsened |
| ic_t_stat | 0.536284 | 0.372664 | -0.163620 | worsened |
| long_short_spread | 0.000676 | 0.005044 | 0.004368 | improved |
| spread_ir | 0.014745 | 0.102188 | 0.087444 | improved |
| spread_t_stat | 0.113256 | 0.784923 | 0.671668 | improved |
| fee_adjusted_return | -0.133774 | 0.083525 | 0.217299 | improved |
| fee_adjusted_sharpe | -0.030813 | 0.052576 | 0.083389 | improved |
| false_safe_tail_rate | 0.095217 | 0.077528 | -0.017689 | improved |
| false_safe_severe_rate | 0.116406 | 0.085773 | -0.030633 | improved |
| severe_downside_recall | 0.883594 | 0.914227 | 0.030633 | improved |
| conservative_bias | -0.167578 | -0.198320 | -0.030742 | more_conservative |
| upside_sacrifice | 0.256570 | 0.291073 | 0.034503 | worsened |

### h4_tide_pvv

| metric | CP113 best val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.012114 | 0.047596 | 0.035482 | improved |
| ic_ir | 0.069818 | 0.335602 | 0.265783 | improved |
| ic_t_stat | 0.536284 | 2.577805 | 2.041521 | improved |
| long_short_spread | 0.000676 | 0.013594 | 0.012918 | improved |
| spread_ir | 0.014745 | 0.353712 | 0.338967 | improved |
| spread_t_stat | 0.113256 | 2.716914 | 2.603658 | improved |
| fee_adjusted_return | -0.133774 | 0.797689 | 0.931463 | improved |
| fee_adjusted_sharpe | -0.030813 | 0.280895 | 0.311707 | improved |
| false_safe_tail_rate | 0.095217 | 0.310330 | 0.215112 | worsened |
| false_safe_severe_rate | 0.116406 | 0.312457 | 0.196052 | worsened |
| severe_downside_recall | 0.883594 | 0.687543 | -0.196052 | worsened |
| conservative_bias | -0.167578 | -0.032620 | 0.134957 | less_conservative |
| upside_sacrifice | 0.256570 | 0.108691 | -0.147878 | improved |

### h6_pvv_patch16_stride8

| metric | CP113 best val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.012114 | -0.044170 | -0.056284 | worsened |
| ic_ir | 0.069818 | -0.255198 | -0.325016 | worsened |
| ic_t_stat | 0.536284 | -1.960211 | -2.496494 | worsened |
| long_short_spread | 0.000676 | -0.008997 | -0.009673 | worsened |
| spread_ir | 0.014745 | -0.152834 | -0.167578 | worsened |
| spread_t_stat | 0.113256 | -1.173938 | -1.287194 | worsened |
| fee_adjusted_return | -0.133774 | -0.533175 | -0.399401 | worsened |
| fee_adjusted_sharpe | -0.030813 | -0.190355 | -0.159542 | worsened |
| false_safe_tail_rate | 0.095217 | 0.101340 | 0.006122 | worsened |
| false_safe_severe_rate | 0.116406 | 0.105818 | -0.010588 | improved |
| severe_downside_recall | 0.883594 | 0.894182 | 0.010588 | improved |
| conservative_bias | -0.167578 | -0.166530 | 0.001048 | less_conservative |
| upside_sacrifice | 0.256570 | 0.273424 | 0.016855 | worsened |

### h8_pvv_patch16_stride8

| metric | CP113 best val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.012114 | -0.040308 | -0.052422 | worsened |
| ic_ir | 0.069818 | -0.267067 | -0.336885 | worsened |
| ic_t_stat | 0.536284 | -2.051381 | -2.587665 | worsened |
| long_short_spread | 0.000676 | -0.022890 | -0.023566 | worsened |
| spread_ir | 0.014745 | -0.343265 | -0.358010 | worsened |
| spread_t_stat | 0.113256 | -2.636668 | -2.749924 | worsened |
| fee_adjusted_return | -0.133774 | -0.794189 | -0.660415 | worsened |
| fee_adjusted_sharpe | -0.030813 | -0.364463 | -0.333650 | worsened |
| false_safe_tail_rate | 0.095217 | 0.114011 | 0.018794 | worsened |
| false_safe_severe_rate | 0.116406 | 0.133490 | 0.017085 | worsened |
| severe_downside_recall | 0.883594 | 0.866510 | -0.017085 | worsened |
| conservative_bias | -0.167578 | -0.149187 | 0.018391 | less_conservative |
| upside_sacrifice | 0.256570 | 0.270880 | 0.014310 | worsened |

### h8_no_fundamentals_patch16_stride8

| metric | CP113 best val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.012114 | 0.019445 | 0.007331 | improved |
| ic_ir | 0.069818 | 0.105712 | 0.035893 | improved |
| ic_t_stat | 0.536284 | 0.811986 | 0.275703 | improved |
| long_short_spread | 0.000676 | -0.004881 | -0.005557 | worsened |
| spread_ir | 0.014745 | -0.071075 | -0.085820 | worsened |
| spread_t_stat | 0.113256 | -0.545939 | -0.659195 | worsened |
| fee_adjusted_return | -0.133774 | -0.409349 | -0.275575 | worsened |
| fee_adjusted_sharpe | -0.030813 | -0.095137 | -0.064324 | worsened |
| false_safe_tail_rate | 0.095217 | 0.119690 | 0.024473 | worsened |
| false_safe_severe_rate | 0.116406 | 0.120713 | 0.004307 | worsened |
| severe_downside_recall | 0.883594 | 0.879287 | -0.004307 | worsened |
| conservative_bias | -0.167578 | -0.149212 | 0.018366 | less_conservative |
| upside_sacrifice | 0.256570 | 0.270524 | 0.013955 | worsened |

### h4_context_light_patch16_stride8

context_light feature_set이 현재 cp63 feature_set plan에 없어 구현하지 않고 design_needed로 기록했다.

## horizon bucket metrics

| candidate | bucket | ic_mean | long_short_spread | false_safe_tail_rate | severe_downside_recall |
| --- | --- | --- | --- | --- | --- |
| h4_pvv_patch16_stride8_repro | h1_h5 | 0.014303 | 0.004068 | 0.043220 | 0.937743 |
| h4_no_fundamentals_patch16_stride8 | h1_h5 | 0.014076 | 0.004044 | 0.072881 | 0.906615 |
| h4_pvv_dense_stride4 | h1_h5 | -0.002551 | 0.001449 | 0.026271 | 0.961089 |
| h4_tide_pvv | h1_h5 | 0.032933 | 0.010627 | 0.252542 | 0.758755 |
| h6_pvv_patch16_stride8 | h1_h5 | 0.000578 | 0.002079 | 0.041525 | 0.937799 |
| h6_pvv_patch16_stride8 | h6_h10 | -0.044170 | -0.008997 | 0.106780 | 0.898182 |
| h8_pvv_patch16_stride8 | h1_h5 | 0.005362 | 0.003773 | 0.058475 | 0.872727 |
| h8_pvv_patch16_stride8 | h6_h10 | -0.016875 | -0.003620 | 0.076271 | 0.931727 |
| h8_no_fundamentals_patch16_stride8 | h1_h5 | 0.011250 | 0.003874 | 0.070339 | 0.903030 |
| h8_no_fundamentals_patch16_stride8 | h6_h10 | -0.013195 | 0.006068 | 0.082203 | 0.927711 |

## 제품 판단

- 기본 후보: `h4_pvv_patch16_stride8_repro`
- h8 visual/risk watch: `[]`
- full_features는 이번 CP에서 기본 후보로 가정하지 않았고 실행하지 않았다. 최소 feature group인 pvv를 우선했다.
- 다음 CP 추천: h4_pvv_patch16_stride8_repro를 같은 CP133 이후 snapshot/hash에서 save-run 전용 재현 CP로 승격한다.
