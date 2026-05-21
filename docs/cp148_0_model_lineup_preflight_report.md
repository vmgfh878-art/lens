# CP148-0-S 모델 라인업 preflight 보고서

판정: **WARN**

이번 CP는 500티커 1D/1W 모델 라인업 실험 전 데이터/피처/학습 환경/TCN quantile baseline 준비 상태를 확인하는 preflight다. DB write, Supabase raw read/write, inference 저장, product run 교체, full training은 수행하지 않았다.

실행 시간은 약 410초였다. 전체 EODHD 500 1D/1W parquet를 full scan하고 tiny forward를 수행했기 때문에 긴 편이며, 다음 preflight 반복 실행 전에는 target/finite scan 캐시 또는 샘플링 모드를 추가하는 것이 좋다.

## 1. 데이터 최신성 및 coverage

| 항목 | rows | tickers | date_min | date_max | duplicate | source | provider |
|---|---:|---:|---|---|---:|---|---|
| price_1D | 1387834 | 503 | 2015-01-02 | 2026-05-05 | 0 | eodhd | eodhd |
| indicators_1D | 1355956 | 503 | 2015-03-30 | 2026-05-05 | 0 | eodhd | eodhd |
| indicators_1W | 258410 | 502 | 2016-02-19 | 2026-05-01 | 0 | eodhd | eodhd |

- adjusted OHLC violation: `0`
- expected provider policy: `eodhd_raw_ohlc_adjusted_close_factor_v3_adjusted_ohlc`

## 2. feature contract

- FEATURE_CONTRACT_VERSION: `v3_adjusted_ohlc`
- MODEL_N_FEATURES: `36`
- source feature 수: `29`
- model feature 수: `36`
- atr_ratio 존재: 1D `True`, 1W `True`
- atr_ratio 모델 feature 포함: `False`
- context_light: `design_needed`

## 3. feature / target sanity

| timeframe | source nonfinite | model nonfinite | target | target nonfinite | has_macro | has_breadth | has_fundamentals |
|---|---:|---:|---|---:|---:|---:|---:|
| 1D | 0 | 0 | target_h5 | 0 | 1.0000 | 0.9523 | 0.0114 |
| 1W | 0 | 0 | target_h4 | 0 | 1.0000 | 1.0000 | 0.0124 |

## 4. split sanity

| split | eligible | excluded | train | val | test | overlap | purge_gap | min_fold |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1D_line_h5 | 473 | 30 | 799739 | 171363 | 171837 | 0 | PASS | PASS |
| 1D_band_h5 | 473 | 30 | 799739 | 171363 | 171837 | 0 | PASS | PASS |
| 1W_line_h4 | 447 | 55 | 122808 | 26346 | 26796 | 0 | PASS | PASS |
| 1W_line_h6 | 447 | 55 | 122808 | 26346 | 26796 | 0 | PASS | PASS |
| 1W_band_h4 | 447 | 55 | 122808 | 26346 | 26796 | 0 | PASS | PASS |

## 5. 모델 forward / loss 계약

| model | line | lower | upper | loss finite | contract |
|---|---|---|---|---|---|
| patchtst | [2, 5] | [2, 5] | [2, 5] | PASS | PASS |
| cnn_lstm | [2, 5] | [2, 5] | [2, 5] | PASS | PASS |
| tide | [2, 5] | [2, 5] | [2, 5] | PASS | PASS |
| tcn_quantile | [2, 5] | [2, 5] | [2, 5] | PASS | PASS |

## 6. cache / ticker registry

- latest 1D feature cache manifest: `ai\cache\features_1D_b841b673f7e6_33f6334e.pt.manifest.json`
- latest 1W feature cache manifest: `ai\cache\features_1W_3a376270bca8_621886c0.pt.manifest.json`
- latest 1D ticker registry: `ai\cache\ticker_id_map_1d_8737d2a1935a.json`, count `473`
- latest 1W ticker registry: `ai\cache\ticker_id_map_1w_9915e57df87e.json`, count `446`

## 7. W&B / Optuna 준비

- W&B는 큰 실험 CP에서 사용자가 VSCode 로컬 터미널에서 직접 켠다.
- CP148-0-S에서는 W&B/Optuna sweep을 실행하지 않았다.
- 명령 템플릿: `$env:WANDB_MODE='online'; C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model patchtst --timeframe 1D --epochs 3 --device cuda --no-compile --wandb --wandb-project lens-cp148`

## 8. 판정

- blockers: `[]`
- warnings: `['context_light feature_set 미정의: design_needed', '1D has_fundamentals coverage 낮음: 0.0114', '1W has_fundamentals coverage 낮음: 0.0124']`
- CP148-LM-1D 바로 실행 가능 여부: `ready`
- CP149-BM-1D 바로 실행 가능 여부: `ready`
- CP150-LM-1W 바로 실행 가능 여부: `ready_with_cache_registry_refresh`
- CP151-BM-1W 바로 실행 가능 여부: `ready_with_cache_registry_refresh`

## 9. 우선 조합

- CP148_LM_1D: PatchTST / TCNQuantile, full_features 또는 no_fundamentals, h5
- CP149_BM_1D: CNN-LSTM / TCNQuantile, price_volatility_volume, h5
- CP150_LM_1W: PatchTST / TiDE / TCNQuantile, price_volatility_volume, h4 우선 h6 보조
- CP151_BM_1W: CNN-LSTM / TiDE / TCNQuantile, price_volatility_volume, h4

## 10. 사람이 판단해야 할 항목

- `context_light`는 현재 feature_set plan에 없으므로 이번 CP에서 만들지 않고 design_needed로 기록했다.
- full_features는 fundamentals coverage 해석 WARN을 유지해야 한다.
- TCNQuantile은 skeleton/tiny forward/loss만 준비됐고, 성능 후보 여부는 다음 smoke에서 판단해야 한다.
- 1W EODHD split eligible은 447개인데 최신 1W ticker registry는 446개이고, 최신 1W feature cache manifest는 yfinance/97 ticker 기준이다. 1W CP150/151은 데이터 자체는 split 가능하지만 EODHD 500 alias/cache/registry를 새로 만들거나 명시 refresh한 뒤 실행해야 한다.
