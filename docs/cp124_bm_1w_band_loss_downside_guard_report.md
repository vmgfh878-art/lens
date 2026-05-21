# CP124-BM 1W Band Loss / Downside Guard 제한 실험

## 1. 결론

- 상태: PASS
- final_recommended_default: `cnn_full_q10_direct_lower_guard_w1p5`
- save-run, DB write, inference 저장, W&B, composite, 프론트 수정은 수행하지 않았다.
- 후보 선택은 validation metric과 validation falling regime metric만 사용했다.
- CP121 기준선은 학습을 재실행하지 않고 checkpoint를 재사용했으며, regime metric은 현재 yfinance local snapshot 기준으로 다시 계산했다.
- test metric은 ai.train 결과 구조상 생성될 수 있으나 read-only count만 기록했고 선택에는 쓰지 않았다.
- lower_breach_penalty/asymmetric_interval 강화는 lower quantile pinball loss weight로만 작게 구현했다.
- width_alignment는 현재 lambda_width가 실제 손실에 연결되지 않아 design_needed로 기록했다.

## 2. 손실 지원 범위

- 기본 ForecastCompositeLoss: line Huber + band pinball + direct 교차 패널티 + direction 보조 손실.
- CP124 추가: `--lower-band-loss-weight`, `--upper-band-loss-weight` CLI. 기본값은 1.0/1.0이라 기존 동작은 유지된다.
- lower guard: lower quantile loss weight 1.5.
- asymmetric guard: lower quantile loss weight 2.0, upper는 1.0 유지.
- width alignment: 큰 구조 변경 없이 지원할 수 없어 이번 CP에서는 실행하지 않았다.

## 3. 실험 요약

| experiment | category | base | runs | gate | test_exp | cov_abs | lower | falling_lower | interval | bw_ic | down_ic | p90_w | failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tide_pvv_q15_param_baseline | baseline_reference | tide_pvv_q15_param | 2 | 1.000000 | 2 | 0.001686 | 0.120173 | 0.242000 | 0.243948 | 0.319981 | 0.017220 | 0.175292 |  |
| tide_pvv_q15_param_lower_guard_w1p5 | experiment_record | tide_pvv_q15_param | 2 | 1.000000 | 2 | 0.002232 | 0.125524 | 0.251759 | 0.249556 | 0.315440 | 0.014854 | 0.178052 | falling_lower_breach_improved |
| tide_pvv_q15_param_asym_guard_w2p0 | experiment_record | tide_pvv_q15_param | 2 | 1.000000 | 2 | 0.000852 | 0.125786 | 0.262880 | 0.234681 | 0.324739 | 0.017175 | 0.162897 | falling_lower_breach_improved |
| tide_pvv_q15_param_width_alignment | design_needed | tide_pvv_q15_param | 0 | 0.000000 | 0 |  |  |  |  |  |  |  | width_alignment_loss_not_implemented |
| cnn_full_q10_direct_baseline | baseline_reference | cnn_full_q10_direct | 2 | 1.000000 | 2 | 0.035536 | 0.073039 | 0.158534 | 0.253210 | 0.247238 | 0.005963 | 0.272726 |  |
| cnn_full_q10_direct_lower_guard_w1p5 | recommended_default | cnn_full_q10_direct | 2 | 1.000000 | 2 | 0.038813 | 0.070418 | 0.155470 | 0.251219 | 0.243598 | 0.004826 | 0.264608 |  |

## 4. Seed별 실행 결과

| experiment | seed | status | exit | gate | cov_abs | lower | interval | bw_ic | down_ic | epoch_s | vram_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tide_pvv_q15_param_baseline | 42 | REUSED_CP121 | 0 | True | 0.001031 | 0.114800 | 0.252183 | 0.306744 | 0.030263 | 6.777200 | 41.510000 |
| tide_pvv_q15_param_baseline | 43 | REUSED_CP121 | 0 | True | 0.002341 | 0.125546 | 0.235712 | 0.333219 | 0.004177 | 6.643967 | 41.510000 |
| tide_pvv_q15_param_lower_guard_w1p5 | 42 | PASS | 0 | True | 0.001118 | 0.129128 | 0.262572 | 0.298803 | 0.024346 | 6.469733 | 41.510000 |
| tide_pvv_q15_param_lower_guard_w1p5 | 43 | PASS | 0 | True | 0.003346 | 0.121920 | 0.236539 | 0.332078 | 0.005361 | 6.558167 | 41.510000 |
| tide_pvv_q15_param_asym_guard_w2p0 | 42 | PASS | 0 | True | 0.000978 | 0.130963 | 0.232924 | 0.319604 | 0.031344 | 6.679967 | 41.510000 |
| tide_pvv_q15_param_asym_guard_w2p0 | 43 | PASS | 0 | True | 0.000725 | 0.120610 | 0.236437 | 0.329874 | 0.003005 | 6.642567 | 41.510000 |
| tide_pvv_q15_param_width_alignment |  | DESIGN_NEEDED |  |  |  |  |  |  |  |  |  |
| cnn_full_q10_direct_baseline | 42 | REUSED_CP121 | 0 | True | 0.067071 | 0.056963 | 0.257448 | 0.263855 | 0.011962 | 20.401767 | 283.950000 |
| cnn_full_q10_direct_baseline | 43 | REUSED_CP121 | 0 | True | 0.004001 | 0.089114 | 0.248971 | 0.230621 | -0.000036 | 23.258200 | 283.950000 |
| cnn_full_q10_direct_lower_guard_w1p5 | 42 | PASS | 0 | True | 0.070479 | 0.052158 | 0.252387 | 0.258109 | 0.010325 | 21.646533 | 283.950000 |
| cnn_full_q10_direct_lower_guard_w1p5 | 43 | PASS | 0 | True | 0.007147 | 0.088677 | 0.250051 | 0.229088 | -0.000674 | 21.669233 | 283.950000 |

## 5. Regime별 Validation 평균

| experiment | regime | samples | cov_abs | lower | interval | bw_ic | down_ic | p90_w |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tide_pvv_q15_param_baseline | high_volatility | 2862.000000 | 0.072292 | 0.152778 | 0.304150 | 0.349932 | 0.024731 | 0.192796 |
| tide_pvv_q15_param_baseline | low_volatility | 2861.000000 | 0.074074 | 0.087600 | 0.183705 | 0.184521 | 0.007670 | 0.155905 |
| tide_pvv_q15_param_baseline | rising | 3520.000000 | 0.009872 | 0.043963 | 0.222367 | 0.339821 | -0.056799 | 0.174385 |
| tide_pvv_q15_param_baseline | falling | 2203.000000 | 0.018055 | 0.242000 | 0.278405 | 0.292395 | 0.112476 | 0.173911 |
| tide_pvv_q15_param_baseline | wide_band | 2862.000000 | 0.009268 | 0.137317 | 0.291328 | 0.305482 | 0.020983 | 0.202711 |
| tide_pvv_q15_param_baseline | narrow_band | 2861.000000 | 0.011028 | 0.103067 | 0.196531 | 0.280587 | -0.003028 | 0.135360 |
| tide_pvv_q15_param_lower_guard_w1p5 | high_volatility | 2862.000000 | 0.073864 | 0.158237 | 0.310237 | 0.343973 | 0.017182 | 0.195767 |
| tide_pvv_q15_param_lower_guard_w1p5 | low_volatility | 2861.000000 | 0.075909 | 0.091882 | 0.188696 | 0.181711 | 0.010073 | 0.159618 |
| tide_pvv_q15_param_lower_guard_w1p5 | rising | 3520.000000 | 0.001562 | 0.045774 | 0.224070 | 0.335403 | -0.035306 | 0.177948 |
| tide_pvv_q15_param_lower_guard_w1p5 | falling | 2203.000000 | 0.005118 | 0.251759 | 0.290074 | 0.288083 | 0.091302 | 0.177538 |
| tide_pvv_q15_param_lower_guard_w1p5 | wide_band | 2862.000000 | 0.013374 | 0.145484 | 0.300151 | 0.300972 | 0.014096 | 0.205123 |
| tide_pvv_q15_param_lower_guard_w1p5 | narrow_band | 2861.000000 | 0.015397 | 0.104640 | 0.198787 | 0.277224 | -0.004614 | 0.138180 |
| tide_pvv_q15_param_asym_guard_w2p0 | high_volatility | 2862.000000 | 0.083997 | 0.160945 | 0.296577 | 0.369634 | 0.021269 | 0.175848 |
| tide_pvv_q15_param_asym_guard_w2p0 | low_volatility | 2861.000000 | 0.080453 | 0.092974 | 0.173252 | 0.184840 | 0.010548 | 0.147930 |
| tide_pvv_q15_param_asym_guard_w2p0 | rising | 3520.000000 | 0.003942 | 0.041903 | 0.208400 | 0.347583 | -0.038236 | 0.161743 |
| tide_pvv_q15_param_asym_guard_w2p0 | falling | 2203.000000 | 0.008852 | 0.262880 | 0.277309 | 0.292801 | 0.102158 | 0.162619 |
| tide_pvv_q15_param_asym_guard_w2p0 | wide_band | 2862.000000 | 0.019794 | 0.144872 | 0.281640 | 0.311158 | 0.019103 | 0.184105 |
| tide_pvv_q15_param_asym_guard_w2p0 | narrow_band | 2861.000000 | 0.016227 | 0.109053 | 0.188195 | 0.288274 | -0.002259 | 0.130467 |
| tide_pvv_q15_param_width_alignment | high_volatility |  |  |  |  |  |  |  |
| tide_pvv_q15_param_width_alignment | low_volatility |  |  |  |  |  |  |  |
| tide_pvv_q15_param_width_alignment | rising |  |  |  |  |  |  |  |
| tide_pvv_q15_param_width_alignment | falling |  |  |  |  |  |  |  |
| tide_pvv_q15_param_width_alignment | wide_band |  |  |  |  |  |  |  |
| tide_pvv_q15_param_width_alignment | narrow_band |  |  |  |  |  |  |  |
| cnn_full_q10_direct_baseline | high_volatility | 2862.000000 | 0.045903 | 0.091850 | 0.305364 | 0.283956 | 0.003370 | 0.284874 |
| cnn_full_q10_direct_baseline | low_volatility | 2861.000000 | 0.089112 | 0.053827 | 0.201090 | 0.137988 | 0.007497 | 0.256040 |
| cnn_full_q10_direct_baseline | rising | 3520.000000 | 0.033807 | 0.019212 | 0.232792 | 0.267963 | -0.070033 | 0.274312 |
| cnn_full_q10_direct_baseline | falling | 2203.000000 | 0.039548 | 0.158534 | 0.285903 | 0.215706 | 0.113498 | 0.270408 |
| cnn_full_q10_direct_baseline | wide_band | 2862.000000 | 0.057704 | 0.060054 | 0.288533 | 0.208615 | -0.006542 | 0.300864 |
| cnn_full_q10_direct_baseline | narrow_band | 2861.000000 | 0.044521 | 0.085634 | 0.217927 | 0.199816 | 0.012103 | 0.208371 |
| cnn_full_q10_direct_lower_guard_w1p5 | high_volatility | 2862.000000 | 0.047956 | 0.090190 | 0.304095 | 0.280109 | 0.001574 | 0.276934 |
| cnn_full_q10_direct_lower_guard_w1p5 | low_volatility | 2861.000000 | 0.091515 | 0.050288 | 0.198313 | 0.134082 | 0.007164 | 0.249576 |
| cnn_full_q10_direct_lower_guard_w1p5 | rising | 3520.000000 | 0.036009 | 0.016903 | 0.231303 | 0.265014 | -0.066862 | 0.266230 |
| cnn_full_q10_direct_lower_guard_w1p5 | falling | 2203.000000 | 0.042612 | 0.155470 | 0.283026 | 0.211538 | 0.105959 | 0.262510 |
| cnn_full_q10_direct_lower_guard_w1p5 | wide_band | 2862.000000 | 0.058927 | 0.057521 | 0.285362 | 0.209456 | -0.006877 | 0.292912 |
| cnn_full_q10_direct_lower_guard_w1p5 | narrow_band | 2861.000000 | 0.047230 | 0.082969 | 0.217052 | 0.194456 | 0.007141 | 0.203957 |

## 6. 판정 기준

- validation band_gate pass.
- coverage_abs_error <= 0.05.
- lower_breach_rate <= 0.18.
- falling regime lower_breach_rate가 같은 baseline보다 개선.
- asymmetric_interval_score가 baseline 대비 10% 초과 악화되지 않음.
- band_width_ic > 0.15.
- downside_width_ic >= 0.

## 7. 산출물

- `docs\cp124_bm_1w_band_loss_downside_guard_report.md`
- `docs\cp124_bm_1w_band_loss_downside_guard_metrics.json`
- `docs\cp124_bm_1w_band_loss_downside_guard_registry.json`
- `docs\cp124_bm_1w_band_loss_downside_guard_summary.csv`
- `docs\cp124_bm_1w_band_loss_downside_guard_logs`
