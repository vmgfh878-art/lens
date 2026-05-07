# CP120-BM 1W Band Mode/Target 제한 실험

## 1. 결론

- 상태: PASS
- save-run, DB write, inference 저장, W&B, composite, Supabase 대량 read, 프론트 수정은 수행하지 않았다.
- yfinance local 1W parquet snapshot만 사용했다.
- recommended_default: `cnn_pvv_q10_direct`
- target probe는 기존 코드가 요청 target을 지원하지 않아 `design_needed`로 기록했다.

## 2. 실행 조건

| 항목 | 값 |
| --- | --- |
| timeframe | 1W |
| horizon | 4 |
| seq_len | 104 |
| epochs | 3 |
| batch_size | 128 |
| checkpoint_selection | band_gate |
| source/provider | yfinance local parquet |
| W&B | disabled |
| save-run | false |

## 3. 실험 결과

| 실험 | model | feature_set | mode | target | 상태 | 분류 | gate | cov_abs | lower | upper | p90_w | interval | bw_ic | down_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_pvv_q10_direct | cnn_lstm | price_volatility_volume | direct | raw_future_return | PASS | recommended_default | True | 0.006014 | 0.080541 | 0.113445 | 0.319798 | 0.301977 | 0.282361 | 0.024778 |
| cnn_full_q10_direct | cnn_lstm | full_features | direct | raw_future_return | PASS | selectable_verified | True | 0.030326 | 0.073024 | 0.096649 | 0.308076 | 0.295776 | 0.274644 | 0.037795 |
| cnn_pvv_q10_param | cnn_lstm | price_volatility_volume | param | raw_future_return | PASS | experiment_record | False | 0.075430 | 0.058419 | 0.066151 | 0.281877 | 0.299695 | 0.291483 | 0.010427 |
| tide_pvv_q15_param | tide | price_volatility_volume | param | raw_future_return | PASS | selectable_verified | True | 0.042784 | 0.157045 | 0.185739 | 0.174758 | 0.291406 | 0.326849 | 0.005363 |
| tide_pvv_q10_param | tide | price_volatility_volume | param | raw_future_return | PASS | experiment_record | True | 0.059751 | 0.107603 | 0.152148 | 0.207530 | 0.321676 | 0.346569 | 0.002856 |
| cnn_pvv_q10_direct_realized_range_probe | cnn_lstm | price_volatility_volume | direct | realized_range | DESIGN_NEEDED | design_needed |  |  |  |  |  |  |  |  |
| cnn_pvv_q10_direct_downside_probe | cnn_lstm | price_volatility_volume | direct | downside_magnitude | DESIGN_NEEDED | design_needed |  |  |  |  |  |  |  |  |

## 4. 검증 기준

| 항목 | 값 |
| --- | --- |
| coverage_abs_error_max | 0.050000 |
| lower_breach_rate_max | 0.180000 |
| asymmetric_interval_score_max | 0.332175 |
| asymmetric_interval_score_reference | 0.301977 |
| band_width_ic_min | 0.150000 |
| downside_width_ic_min | 0.000000 |
| p90_band_width_max | 0.367768 |
| p90_reference_source | CP119 pvv_q10_direct p90 * 1.15 |

## 5. Direct / Param 해석

- direct 기준선은 `cnn_pvv_q10_direct`다. 이 후보는 CP119 recommended_default를 CP120에서 재현하는 기준선이다.
- `cnn_full_q10_direct`는 기준을 통과했다. PVV q10보다 coverage_abs_error는 크지만 interval, p90 width, downside_width_ic는 더 좋아 selectable_verified 대안으로 남긴다.
- `cnn_pvv_q10_param`은 같은 PVV/q10/raw target 조건에서 interval과 band_width_ic는 좋지만 band_gate fail, coverage_abs_error 0.075430으로 raw 제품 후보가 아니다.
- `tide_pvv_q15_param`은 기준을 통과해 TiDE 1W BM 대안으로 남긴다. 다만 upper breach 0.185739와 downside_width_ic 0.005363은 default가 되기 어려운 이유다.
- `tide_pvv_q10_param`은 band_width_ic가 가장 높지만 coverage_abs_error 0.059751로 기준을 넘겨 experiment_record다. TiDE에서 q10 확장은 coverage/width 균형이 깨졌다.

## 6. Target Probe 해석

- 현재 지원 target: `raw_future_return, market_excess_return, volatility_normalized_return, direction_label, rank_target`
- 요청된 `realized_range`, `realized_volatility`, `downside_magnitude`, `tail_event_probability`는 기존 target 경로에 없다.
- 큰 target 구조 변경은 이번 CP 금지 범위이므로 구현하지 않았다.

## 7. Candidate Registry

| category | display_name | feature_set | mode | target | run_id | strength | weakness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| recommended_default | 1W cnn_lstm cnn_pvv_q10_direct | price_volatility_volume | direct | raw_future_return | cnn_lstm-1W-57b002032e33 | coverage_abs_error=0.006014, interval=0.301977, band_width_ic=0.282361 | 검증 기준 통과 |
| selectable_verified | 1W cnn_lstm cnn_full_q10_direct | full_features | direct | raw_future_return | cnn_lstm-1W-62647a758c06 | coverage_abs_error=0.030326, interval=0.295776, band_width_ic=0.274644 | PVV q10보다 coverage_abs_error가 큼 |
| experiment_record | 1W cnn_lstm cnn_pvv_q10_param | price_volatility_volume | param | raw_future_return | cnn_lstm-1W-34ab90f3acdd | coverage_abs_error=0.075430, interval=0.299695, band_width_ic=0.291483 | band_gate fail 및 coverage_abs_error 0.075430 |
| selectable_verified | 1W tide tide_pvv_q15_param | price_volatility_volume | param | raw_future_return | tide-1W-600c64d3e10b | coverage_abs_error=0.042784, interval=0.291406, band_width_ic=0.326849 | upper breach 0.185739와 낮은 downside_width_ic 주의 |
| experiment_record | 1W tide tide_pvv_q10_param | price_volatility_volume | param | raw_future_return | tide-1W-5eb9c38911f6 | coverage_abs_error=0.059751, interval=0.321676, band_width_ic=0.346569 | coverage_abs_error 0.059751로 기준 초과 |
| design_needed | 1W cnn_lstm cnn_pvv_q10_direct_realized_range_probe | price_volatility_volume | direct | realized_range |  | 기존 target 경로 미지원으로 학습 미실행 | target 설계/구현 필요 |
| design_needed | 1W cnn_lstm cnn_pvv_q10_direct_downside_probe | price_volatility_volume | direct | downside_magnitude |  | 기존 target 경로 미지원으로 학습 미실행 | target 설계/구현 필요 |

## 8. 산출물

- `docs\cp120_bm_1w_band_mode_target_experiment_report.md`
- `docs\cp120_bm_1w_band_mode_target_experiment_metrics.json`
- `docs\cp120_bm_1w_band_candidate_registry.json`
- `docs\cp120_bm_1w_band_mode_target_summary.csv`
- `docs\cp120_bm_1w_band_mode_target_experiment_logs`
