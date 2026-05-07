# CP121-BM 1W Band Verified 후보 안정성 검증

## 1. 결론

- 상태: PASS
- save-run, DB write, inference 저장, W&B, composite, Supabase 대량 read, 프론트 수정은 수행하지 않았다.
- 후보 선정은 validation metric과 validation regime metric만 사용했다.
- `ai.train` 결과 JSON 특성상 test_metrics는 생성되었지만 read-only count만 기록했고 후보 선정에는 쓰지 않았다.
- recommended_default: `tide_pvv_q15_param`

## 2. 후보별 안정성 요약

| candidate | category | runs | gate | test_exposure | cov_mean | cov_std | lower_mean | interval_mean | interval_std | bw_ic_mean | down_ic_mean | p90_mean | failures | regime_warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_pvv_q10_direct | unstable_verified | 2 | 0.500000 | 2 | 0.085571 | 0.011292 | 0.041739 | 0.265048 | 0.006336 | 0.265318 | 0.018944 | 0.328967 | band_gate_mostly_pass,validation_coverage_abs_error |  |
| cnn_full_q10_direct | selectable_verified | 2 | 1.000000 | 2 | 0.035536 | 0.031535 | 0.073039 | 0.253210 | 0.004239 | 0.247238 | 0.005963 | 0.272726 |  | narrow_band coverage_abs_error=0.278534,falling lower_breach_rate=0.275999,narrow_band band_width_ic=0.047915 |
| tide_pvv_q15_param | recommended_default | 2 | 1.000000 | 2 | 0.001686 | 0.000655 | 0.120173 | 0.243948 | 0.008236 | 0.319981 | 0.017220 | 0.175292 |  | falling lower_breach_rate=0.282671 |

## 3. Seed별 Validation 결과

| candidate | seed | status | gate | run_id | cov_abs | lower | interval | bw_ic | down_ic | p90_w |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_pvv_q10_direct | 42 | PASS | True | cnn_lstm-1W-5227f9eb6457 | 0.074279 | 0.045038 | 0.258712 | 0.286369 | 0.024693 | 0.322726 |
| cnn_pvv_q10_direct | 43 | PASS | False | cnn_lstm-1W-e77f287a7350 | 0.096864 | 0.038441 | 0.271384 | 0.244268 | 0.013196 | 0.335208 |
| cnn_full_q10_direct | 42 | PASS | True | cnn_lstm-1W-194c9afd040a | 0.067071 | 0.056963 | 0.257448 | 0.263855 | 0.011962 | 0.308116 |
| cnn_full_q10_direct | 43 | PASS | True | cnn_lstm-1W-0a032a22ce39 | 0.004001 | 0.089114 | 0.248971 | 0.230621 | -0.000036 | 0.237336 |
| tide_pvv_q15_param | 42 | PASS | True | tide-1W-88a3e1b4ae95 | 0.001031 | 0.114800 | 0.252183 | 0.306744 | 0.030263 | 0.188959 |
| tide_pvv_q15_param | 43 | PASS | True | tide-1W-8bd52bfdf5fd | 0.002341 | 0.125546 | 0.235712 | 0.333219 | 0.004177 | 0.161625 |

## 4. Regime 평가 방식

- high/low volatility: validation raw h4 realized volatility의 median split.
- rising/falling: validation h4 최종 raw return 부호.
- wide/narrow band: 예측 band 평균 폭의 median split.
- regime metric은 후보 선정 보조 안정성 확인용이며 test set은 사용하지 않았다.

## 5. Regime별 Validation 평균

| candidate | regime | samples | cov_abs | lower | interval | bw_ic | down_ic | p90_w |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_pvv_q10_direct | high_volatility | 12341.000000 | 0.037213 | 0.079633 | 0.334647 | 0.358548 | 0.039789 | 0.371969 |
| cnn_pvv_q10_direct | low_volatility | 12339.000000 | 0.056188 | 0.066122 | 0.251676 | 0.159796 | 0.030074 | 0.342004 |
| cnn_pvv_q10_direct | rising | 14170.000000 | 0.016840 | 0.029402 | 0.281282 | 0.309799 | -0.160513 | 0.356586 |
| cnn_pvv_q10_direct | falling | 10510.000000 | 0.075963 | 0.131494 | 0.309186 | 0.264752 | 0.244605 | 0.357576 |
| cnn_pvv_q10_direct | wide_band | 12341.000000 | 0.040029 | 0.073505 | 0.331914 | 0.265025 | 0.027810 | 0.388988 |
| cnn_pvv_q10_direct | narrow_band | 12339.000000 | 0.044007 | 0.072251 | 0.254410 | 0.234715 | 0.032851 | 0.317203 |
| cnn_full_q10_direct | high_volatility | 12341.000000 | 0.226059 | 0.155782 | 0.450278 | 0.146669 | -0.010798 | 0.305452 |
| cnn_full_q10_direct | low_volatility | 12339.000000 | 0.149096 | 0.129609 | 0.318513 | 0.053593 | -0.008614 | 0.297494 |
| cnn_full_q10_direct | rising | 14170.000000 | 0.189106 | 0.043825 | 0.343338 | 0.119012 | 0.087211 | 0.301176 |
| cnn_full_q10_direct | falling | 10510.000000 | 0.201534 | 0.275999 | 0.439763 | 0.100935 | -0.082923 | 0.301963 |
| cnn_full_q10_direct | wide_band | 12341.000000 | 0.155245 | 0.107710 | 0.363428 | 0.110763 | -0.027422 | 0.309612 |
| cnn_full_q10_direct | narrow_band | 12339.000000 | 0.278534 | 0.177689 | 0.405376 | 0.047915 | -0.000772 | 0.288043 |
| tide_pvv_q15_param | high_volatility | 12341.000000 | 0.073319 | 0.181975 | 0.307248 | 0.382671 | 0.055402 | 0.176143 |
| tide_pvv_q15_param | low_volatility | 12339.000000 | 0.079399 | 0.119408 | 0.186486 | 0.182553 | 0.037114 | 0.151038 |
| tide_pvv_q15_param | rising | 14170.000000 | 0.021277 | 0.052805 | 0.214278 | 0.349032 | -0.085522 | 0.162516 |
| tide_pvv_q15_param | falling | 10510.000000 | 0.021230 | 0.282671 | 0.290816 | 0.302609 | 0.169013 | 0.164539 |
| tide_pvv_q15_param | wide_band | 12341.000000 | 0.028944 | 0.168270 | 0.297387 | 0.294610 | 0.052450 | 0.183894 |
| tide_pvv_q15_param | narrow_band | 12339.000000 | 0.035017 | 0.133115 | 0.196348 | 0.277553 | 0.031269 | 0.138085 |

## 6. Candidate Registry

| category | display_name | feature_set | mode | test_exposure | runs | strength | weakness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unstable_verified | 1W cnn_lstm cnn_pvv_q10_direct | price_volatility_volume | direct | 2 | cnn_lstm-1W-5227f9eb6457,cnn_lstm-1W-e77f287a7350 | val cov_abs mean=0.085571, val interval mean=0.265048, val bw_ic mean=0.265318 | band_gate_mostly_pass, validation_coverage_abs_error |
| selectable_verified | 1W cnn_lstm cnn_full_q10_direct | full_features | direct | 2 | cnn_lstm-1W-194c9afd040a,cnn_lstm-1W-0a032a22ce39 | val cov_abs mean=0.035536, val interval mean=0.253210, val bw_ic mean=0.247238 | 검증 기준 통과; regime warning: narrow_band coverage_abs_error=0.278534, falling lower_breach_rate=0.275999, narrow_band band_width_ic=0.047915 |
| recommended_default | 1W tide tide_pvv_q15_param | price_volatility_volume | param | 2 | tide-1W-88a3e1b4ae95,tide-1W-8bd52bfdf5fd | val cov_abs mean=0.001686, val interval mean=0.243948, val bw_ic mean=0.319981 | 검증 기준 통과; regime warning: falling lower_breach_rate=0.282671 |

## 7. 산출물

- `docs\cp121_bm_1w_band_verified_stability_report.md`
- `docs\cp121_bm_1w_band_verified_stability_metrics.json`
- `docs\cp121_bm_1w_band_verified_registry.json`
- `docs\cp121_bm_1w_band_verified_stability_summary.csv`
- `docs\cp121_bm_1w_band_verified_stability_logs`
