# CP103-D yfinance 신규 거래일 append 운영 게이트

생성일: 2026-05-04

## 1. Executive Summary

최종 판정: WARN_NO_NEW_MARKET_DAY

yfinance가 snapshot 최신일보다 새로운 거래일 row를 아직 제공하지 않았다. local append와 indicator incremental refresh는 다음 거래일 데이터 반영 후 재검증해야 한다.

이번 CP는 EODHD 해지 전 마지막 운영 gate로, yfinance가 local snapshot 최신일보다 새로운 1D row를 주는지 확인하고, 새 row가 있을 때 local parquet append, 1D indicator incremental refresh, 제품 inference, Supabase latest-only thin upload, API 조회까지 이어지는지 검증한다.

## 2. 환경

| 항목 | 값 |
|---|---|
| provider | yfinance |
| source | yfinance |
| timeframe | 1D |
| horizon | 5 |
| tickers | AAPL, MSFT, NVDA, TSLA, NFLX |
| line run | patchtst-1D-efad3c29d803 |
| band run | cnn_lstm-1D-d0c780dee5e8 |
| fallback provider | None |
| EODHD key present | False |

## 3. Snapshot 현재 상태

| 항목 | Before | After |
|---|---:|---:|
| price latest date | 2026-05-01 | 2026-05-01 |
| indicator latest date | 2026-05-01 | 2026-05-01 |
| price rows | 284900 | 284900 |
| indicator rows | 279000 | 279000 |
| price ticker count | 100 | 100 |
| indicator ticker count | 100 | 100 |
| source_data_hash | 3e4ee198 | 3e4ee198 |

## 4. yfinance 최신 조회

요청:
- start_date: 2026-04-24
- end_date: 2026-05-04
- fallback_used: False
- successful_fetch_count: 5
- failed_or_empty_fetch_count: 0

| ticker | status | fetched rows | new rows | fetched min | fetched max | fallback | adjusted contract |
|---|---:|---:|---:|---|---|---:|---:|
| AAPL | PASS | 6 | 0 | 2026-04-24 | 2026-05-01 | False | True |
| MSFT | PASS | 6 | 0 | 2026-04-24 | 2026-05-01 | False | True |
| NFLX | PASS | 6 | 0 | 2026-04-24 | 2026-05-01 | False | True |
| NVDA | PASS | 6 | 0 | 2026-04-24 | 2026-05-01 | False | True |
| TSLA | PASS | 6 | 0 | 2026-04-24 | 2026-05-01 | False | True |

## 5. Local Parquet Append

| 항목 | 값 |
|---|---:|
| attempted | True |
| price_rows_written | 0 |
| reason | no_new_rows |
| indicator_rebuild_tickers |  |
| indicator_rows_after_rebuild | None |

신규 row가 없으면 이 CP는 `WARN_NO_NEW_MARKET_DAY`로 종료한다. 이는 구조 문제라기보다 Yahoo Finance가 아직 snapshot 최신일보다 새 거래일 row를 제공하지 않았다는 의미다.

## 6. Indicator Refresh 및 Feature Gate

| 항목 | 값 |
|---|---:|
| snapshot_gate_status | PASS |
| MODEL_N_FEATURES | 36 |
| atr_ratio_in_MODEL_FEATURE_COLUMNS | False |
| feature_non_finite_count | 0 |
| target_non_finite_count | 0 |

## 7. 제품 Inference 및 Thin Upload

| 항목 | 값 |
|---|---:|
| inference attempted | False |
| prediction_rows | None |
| evaluation_rows | None |
| all_forecast_horizon_5 | None |
| all_series_length_5 | None |
| lower_lte_upper | None |
| thin_upload_attempted | None |
| prediction_rows_written | None |
| evaluation_rows_written | None |

## 8. API 확인

| 항목 | 값 |
|---|---|
| attempted | None |
| AAPL line found | None |
| AAPL band found | None |
| line asof_date | None |
| band asof_date | None |
| expected latest asof_date | None |
| asof_date matched | None |

## 9. 금지 항목 확인

| 항목 | 발생 |
|---|---:|
| EODHD API call | False |
| EODHD fallback | False |
| Supabase price_data/indicators bulk read | False |
| Supabase price_data/indicators write | False |
| full model training | False |
| 1W/1M processing | False |
| composite save | False |
| DB row delete | False |
| cache/checkpoint delete | False |

## 10. 해지 전 판단

- PASS이면 EODHD 해지 가능 후보로 본다.
- WARN_NO_NEW_MARKET_DAY이면 해지 판정은 보류하되 구조 문제는 아니다. 다음 거래일 데이터가 yfinance에 반영된 뒤 같은 명령을 재실행한다.
- FAIL이면 EODHD 해지 보류다.

현재 판정: WARN_NO_NEW_MARKET_DAY

## 11. 실행 명령

```powershell
$env:MARKET_DATA_PROVIDER="yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER=""
$env:LENS_DATA_BACKEND="local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS="1"
$env:LENS_LOCAL_SNAPSHOT_DIR="C:\Users\user\lens\data\parquet"
python scripts\cp103_yfinance_new_day_append_gate.py
```
