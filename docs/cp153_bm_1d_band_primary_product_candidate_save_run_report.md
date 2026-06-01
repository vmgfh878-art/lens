# CP153-BM 1D Band Primary Product Candidate Local Save-Run Report

## 결론

- 판정: `1D band primary product candidate ready for DB attach`
- DB write는 수행하지 않았다.
- DB attach는 Supabase quota refill 예정일인 2026-05-16 이후 사용자 승인 전까지 금지한다.
- 이번 저장 대상은 primary `tide_s60_q15_param + lower_focused`뿐이며 TCN은 저장하지 않았다.

## 계약

- provider/source: `yfinance / yfinance`
- source_data_hash: `90666b44cbfb8e5c`
- timeframe/horizon: `1D / h5`
- feature_version: `v3_adjusted_ohlc`
- feature_set: `price_volatility_volume`
- feature_columns: `log_return, open_ratio, high_ratio, low_ratio, vol_change, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position`
- target: `raw_future_return`
- model_role/output_role: `band / band`
- q_low/q_high: `0.15 / 0.85`
- calibration: `lower_focused` validation-only fit

## Save-Run Test Metrics

| metric | save-run test | Stage 5T Tide ref | delta |
| --- | ---: | ---: | ---: |
| coverage_abs_error | 0.009887 | 0.025440 | -0.015553 |
| lower_breach_rate | 0.158586 | 0.142540 | 0.016046 |
| upper_breach_rate | 0.151300 | 0.156973 | -0.005673 |
| band_width_ic | 0.375986 | 0.373995 | 0.001991 |
| downside_width_ic | 0.086588 | 0.086193 | 0.000395 |

## Product Gate

- gate pass: `True`
- reasons: `none`

## Latest-Only Artifact

- artifact path: `C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_latest_predictions.json`
- jsonl path: `C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_latest_predictions.jsonl`
- asof_date: `2026-05-08`
- prediction_count: `474`
- contract status: `PASS`
- line payload absent: `True`
- composite payload absent: `True`

## Product Payload Samples

- `A` lower=[113.7789306640625, 113.94703674316406, 113.25983428955078, 112.43943786621094, 111.95502471923828] upper=[117.73162078857422, 119.18350982666016, 119.57622528076172, 120.1784439086914, 120.32289123535156]
- `AAPL` lower=[290.0047607421875, 287.2141418457031, 285.18658447265625, 283.6432189941406, 283.57330322265625] upper=[298.85418701171875, 299.3099060058594, 299.1087341308594, 300.700927734375, 302.60260009765625]
- `ABBV` lower=[197.989990234375, 196.5539093017578, 196.91595458984375, 195.51544189453125, 196.17788696289062] upper=[205.3247833251953, 205.97193908691406, 208.82147216796875, 209.21861267089844, 211.9501495361328]
- `ABNB` lower=[138.55807495117188, 138.07276916503906, 138.9881591796875, 136.39276123046875, 136.215576171875] upper=[144.48464965820312, 146.43055725097656, 148.45877075195312, 148.5532684326172, 149.16036987304688]
- `ABT` lower=[83.01233673095703, 82.42610931396484, 82.34636688232422, 81.99369812011719, 81.35452270507812] upper=[86.17831420898438, 86.68638610839844, 87.566162109375, 88.48983764648438, 88.26963806152344]

## 산출물

- metrics: `C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_save_run_metrics.json`
- run meta: `C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_run_meta.json`
- calibration params: `C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_calibration_params.json`
- latest artifact: `C:\Users\user\lens\docs\cp153_bm_1d_band_primary_product_candidate_latest_predictions.json`
- logs: `C:\Users\user\lens\docs\cp153_bm_1d_band_primary_save_run_logs`
