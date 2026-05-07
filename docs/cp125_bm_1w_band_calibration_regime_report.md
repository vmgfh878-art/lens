# CP125-BM 1W Band Calibration / Regime Final Evaluation

## 1. 결론

- 상태: PASS
- 기본 후보: `cnn_full_q10_direct_lower_guard_w1p5`
- 대안 후보: `None`
- 저장 전 추가 확인 필요: True
- save-run, DB write, inference 저장, W&B, composite, 프론트 수정, 새 후보 탐색, 새 target 구현은 수행하지 않았다.
- test split은 CP125에서 새로 열지 않았고, 이전 CP의 test_exposure_count만 registry에 이월했다.
- raw 평가는 전체 validation 기준이고, calibration 평가는 validation 날짜 앞 절반 fit / 뒤 절반 eval 기준이다.

## 2. 후보 요약

| candidate | category | raw_cov | raw_lower | raw_falling | raw_high_vol | raw_high_atr | raw_interval | raw_p90 | raw_bw_ic | raw_down_ic | cal_cov | cal_lower | cal_falling | cal_mult | cal_width_inc | raw_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_full_q10_direct_lower_guard_w1p5 | recommended_default | 0.038813 | 0.070418 | 0.155867 | 0.090365 | 0.077131 | 0.251219 | 0.264610 | 0.243599 | 0.004833 | 0.046048 | 0.126847 | 0.270468 | 0.796941 | -0.155877 |  |
| tide_pvv_q15_param | unstable_or_rejected | 0.002407 | 0.119933 | 0.241262 | 0.152734 | 0.152865 | 0.243900 | 0.175292 | 0.320026 | 0.017589 | 0.028780 | 0.174313 | 0.326839 | 0.890718 | -0.049731 | raw_falling_lower_breach_rate |

## 3. Raw / Calibration 분리 원칙

- raw 기준 성능은 모델 원본 band 자체의 성능으로 해석한다.
- scalar calibration은 validation 앞 절반에서 fit하고 뒤 절반에서만 평가했다.
- calibration으로 개선된 수치는 raw 모델 성능으로 해석하지 않는다.
- calibration 없이는 regime 기준을 넘지 못하는 후보는 기본 후보가 아니라 calibration_only_candidate 이하로 분류한다.

## 4. Candidate Registry

| category | display_name | feature_set | mode | test_exp | why_not_default |
| --- | --- | --- | --- | --- | --- |
| recommended_default | 1W CNN-LSTM full q10 direct lower guard | full_features | direct | 2 | 현재 기본 후보 |
| unstable_or_rejected | 1W TiDE PVV q15 param | price_volatility_volume | param | 2 | raw_falling_lower_breach_rate |

## 5. Regime CSV

- regime별 상세 수치는 `docs\cp125_bm_1w_band_regime_summary.csv`에 저장했다.

## 6. 산출물

- `docs\cp125_bm_1w_band_calibration_regime_report.md`
- `docs\cp125_bm_1w_band_calibration_regime_metrics.json`
- `docs\cp125_bm_1w_band_final_registry.json`
- `docs\cp125_bm_1w_band_regime_summary.csv`
