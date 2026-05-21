# CP119-BM 1W Band Feature Group 제한 실험

## 1. 결론

- 상태: PASS
- 새 save-run, DB write, inference 저장, W&B, composite, Supabase 대량 read, 프론트 수정은 수행하지 않았다.
- yfinance local 1W parquet snapshot만 사용했다.
- recommended_default: `pvv_q10_direct`
- 검증 기준을 통과하지 못한 watch/research 성격 후보는 제품 UI 후보가 아니라 experiment_record로만 남겼다.

## 2. 실행 조건

| 항목 | 값 |
| --- | --- |
| timeframe | 1W |
| horizon | 4 |
| seq_len | 104 |
| model | cnn_lstm |
| band_mode | direct |
| epochs | 3 |
| batch_size | 128 |
| checkpoint_selection | band_gate |
| W&B | disabled |
| save-run | false |

## 3. 실험 결과

| 실험 | feature_set | q | 상태 | 분류 | gate | cov_abs | lower | upper | p90_w | interval | bw_ic | down_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pvv_q10_direct | price_volatility_volume | 0.100000 | PASS | recommended_default | True | 0.006014 | 0.080541 | 0.113445 | 0.319798 | 0.301977 | 0.282361 | 0.024778 |
| pvv_q15_direct | price_volatility_volume | 0.150000 | PASS | selectable_verified | True | 0.015765 | 0.135653 | 0.148582 | 0.333164 | 0.317112 | 0.208742 | 0.020930 |
| no_fundamentals_q10_direct | no_fundamentals | 0.100000 | PASS | experiment_record | True | 0.003608 | 0.069759 | 0.126632 | 0.391806 | 0.365791 | 0.112298 | 0.020429 |
| technical_only_q10_direct | technical_only | 0.100000 | PASS | selectable_verified | True | 0.006014 | 0.080541 | 0.113445 | 0.319798 | 0.301977 | 0.282361 | 0.024778 |
| full_features_q10_direct | full_features | 0.100000 | PASS | selectable_verified | True | 0.030326 | 0.073024 | 0.096649 | 0.308076 | 0.295776 | 0.274644 | 0.037795 |
| price_return_only_q10_direct | price_return_only | 0.100000 | PASS | experiment_record | False | 0.040893 | 0.086598 | 0.072509 | 0.288666 | 0.298979 | 0.247211 | 0.025051 |

## 4. 검증 기준

| 항목 | 값 |
| --- | --- |
| coverage_abs_error_max | 0.050000 |
| lower_breach_rate_max | 0.180000 |
| asymmetric_interval_score_max | 0.317112 |
| band_width_ic_min | 0.150000 |
| downside_width_ic_min | 0.000000 |
| p90_band_width_max | 0.383139 |
| p90_reference_source | CP113 q15 PVV p90 * 1.15 |

## 5. Candidate Registry

| category | display_name | feature_set | run_id | strength | weakness |
| --- | --- | --- | --- | --- | --- |
| recommended_default | 1W CNN-LSTM pvv_q10_direct | price_volatility_volume | cnn_lstm-1W-7ab0f4154c12 | coverage_abs_error=0.006014, interval=0.301977, band_width_ic=0.282361 | 검증 기준 통과 |
| selectable_verified | 1W CNN-LSTM pvv_q15_direct | price_volatility_volume | cnn_lstm-1W-ca25f4d5bf9d | coverage_abs_error=0.015765, interval=0.317112, band_width_ic=0.208742 | PVV q10보다 coverage_abs_error와 lower/upper breach가 큼 |
| experiment_record | 1W CNN-LSTM no_fundamentals_q10_direct | no_fundamentals | cnn_lstm-1W-be65936ee0f3 | coverage_abs_error=0.003608, interval=0.365791, band_width_ic=0.112298 | asymmetric_interval_score, band_width_ic, p90_band_width |
| selectable_verified | 1W CNN-LSTM technical_only_q10_direct | technical_only | cnn_lstm-1W-529429c471e4 | coverage_abs_error=0.006014, interval=0.301977, band_width_ic=0.282361 | price_volatility_volume와 동일 11개 컬럼이라 별도 UI 항목으로는 중복 |
| selectable_verified | 1W CNN-LSTM full_features_q10_direct | full_features | cnn_lstm-1W-d394871132bc | coverage_abs_error=0.030326, interval=0.295776, band_width_ic=0.274644 | PVV q10보다 coverage_abs_error가 큼 |
| experiment_record | 1W CNN-LSTM price_return_only_q10_direct | price_return_only | cnn_lstm-1W-7a488d444f66 | coverage_abs_error=0.040893, interval=0.298979, band_width_ic=0.247211 | band_gate_pass |

## 6. 해석

- `price_volatility_volume` q10은 CP114 기준 후보 재현성 확인 대상이다. 새 실험이 기준을 넘으면 recommended_default를 새 run으로 갱신한다.
- `price_volatility_volume` q15는 q10/q90이 과보수일 때의 비교 대안이다. 기준을 넘더라도 q10보다 균형이 약하면 selectable_verified로 둔다.
- `technical_only`는 현재 CP63 정의상 `price_volatility_volume`과 같은 11개 컬럼이라 metric도 동일하다. 검증 기준은 통과했지만 별도 UI 항목으로는 중복이므로 PVV로 통합 관리하는 편이 낫다.
- `full_features` q10은 검증 기준을 통과했다. PVV보다 coverage_abs_error는 크지만 interval, p90 width, downside_width_ic는 더 좋아 selectable_verified 대안으로 남긴다.
- `no_fundamentals`는 coverage만 보면 좋지만 interval, dynamic width, p90 width 기준을 놓쳐 experiment_record다.
- `price_return_only`는 metric 일부가 좋지만 band_gate를 통과하지 못해 제품 UI 후보가 아니다.
- calibration 전제 후보는 verified로 올리지 않는다. 이번 판정은 raw band 기준이다.

## 7. 산출물

- `docs\cp119_bm_1w_band_feature_group_experiment_report.md`
- `docs\cp119_bm_1w_band_feature_group_experiment_metrics.json`
- `docs\cp119_bm_1w_band_candidate_registry.json`
- `docs\cp119_bm_1w_band_experiment_summary.csv`
- `docs\cp119_bm_1w_band_feature_group_experiment_logs`
