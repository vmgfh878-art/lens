# CP115-D yfinance 완료 거래일 append + 1D indicator refresh 리허설

생성일: 2026-05-05

## 1. Executive Summary

최종 판정: PASS

완료 거래일만 local price parquet에 append했고 5티커 1D indicator refresh까지 통과했다.

이번 CP는 DB/Supabase를 쓰지 않고 local parquet만 업데이트했다. 모델 학습, inference, thin upload, checkpoint load, torch import는 수행하지 않았다.

## 2. Append 전후 Snapshot

| 항목 | Before | After |
|---|---:|---:|
| 1D price latest date | 2026-05-01 | 2026-05-04 |
| 1D price rows | 284900 | 284905 |
| 1D indicator latest date | 2026-05-01 | 2026-05-04 |
| 1D indicator rows | 279000 | 279005 |
| price duplicate ticker/date/source | 0 | 0 |
| indicator duplicate ticker/timeframe/date/source | 0 | 0 |

## 3. Backup

| 항목 | 값 |
|---|---|
| created | True |
| price backup | data\parquet\backups\price_data_yfinance_before_cp115_20260505_233938.parquet |
| indicator backup | data\parquet\backups\indicators_yfinance_1D_before_cp115_20260505_233938.parquet |

## 4. yfinance Fetch 및 Partial 제외

요청:
- tickers: AAPL, MSFT, NVDA, TSLA, NFLX
- snapshot_latest_date: 2026-05-01
- current_date: 2026-05-05
- start_date: 2026-04-17
- end_date: 2026-05-05
- completed row rule: row.date < current_date

| ticker | status | fetched rows | new rows | appended rows | appended dates | partial excluded dates | fallback | adjusted OHLC |
|---|---:|---:|---:|---:|---|---|---:|---:|
| AAPL | PASS | 13 | 2 | 1 | ['2026-05-04'] | ['2026-05-05'] | False | True |
| MSFT | PASS | 13 | 2 | 1 | ['2026-05-04'] | ['2026-05-05'] | False | True |
| NFLX | PASS | 13 | 2 | 1 | ['2026-05-04'] | ['2026-05-05'] | False | True |
| NVDA | PASS | 13 | 2 | 1 | ['2026-05-04'] | ['2026-05-05'] | False | True |
| TSLA | PASS | 13 | 2 | 1 | ['2026-05-04'] | ['2026-05-05'] | False | True |

partial 제외:
- excluded_partial_dates: ['2026-05-05']
- expected_append_dates: ['2026-05-04']
- appended_dates: ['2026-05-04']

## 5. Price Append 검증

| 항목 | 값 |
|---|---:|
| price_rows_appended | 5 |
| expected_price_rows_appended | 5 |
| appended_ticker_count | 5 |
| duplicate_after_append | 0 |
| adjusted_ohlc_passed | True |
| adjusted_ohlc_violations | [] |

## 6. 1D Indicator Incremental Refresh

| 항목 | 값 |
|---|---:|
| rebuilt_tickers | ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'NFLX'] |
| rebuilt_rows | 13955 |
| rebuilt_date_min | 2015-03-30 |
| rebuilt_date_max | 2026-05-04 |
| combined_rows | 279005 |
| duplicate_ticker_timeframe_date_source | 0 |
| source_values | ['yfinance'] |
| provider_values | ['yfinance'] |

## 7. Source/Hash/Cache 변화

| 항목 | 값 |
|---|---|
| price sha before | 3a04a0c4848523368a6aa7fa2455014bb7b5086848952222b570d48726dfbd9d |
| price sha after | 6f6527c791a2db96c5c8bc52c07be4ac7900374d8a2c0c47ac018eb9399efad7 |
| price hash changed | True |
| indicator sha before | 2f00981d4d8f8f472495de85905c74a3d8e6066df3a555685996959067a0db8a |
| indicator sha after | ffce024591d57c5a2730101757ac03d2110352097e1242e8195556309db3f871 |
| indicator hash changed | True |
| feature/index cache touched | False |

주의: 이번 CP는 torch import와 `ai.preprocessing` import가 금지되어 있어서 모델 feature/index cache를 생성하지 않았다. 대신 local parquet file hash 변화로 source snapshot 변화만 기록했다.

## 8. Feature/Target Finite

| 항목 | 값 |
|---|---:|
| feature_rows_checked | 13955 |
| feature_columns_checked | 29 |
| feature_non_finite_count | 0 |
| target_horizon | 5 |
| target_rows_checked | 14225 |
| target_non_finite_count | 0 |
| passed | True |

## 9. EODHD 해지 가능 여부 업데이트

데이터 append/refresh gate는 PASS다. CP101의 EODHD-off 제품 루프 통과 이력과 결합하면 EODHD 해지 가능 후보로 볼 수 있다. 다만 이번 CP에서는 inference/thin upload가 금지였으므로, 해지 직전에는 최신 asof 기준 제품 루프를 한 번 더 얇게 확인하는 것이 안전하다.

## 10. 금지 항목 확인

| 항목 | 발생 |
|---|---:|
| DB write | False |
| Supabase price_data/indicators 대량 read/write | False |
| 모델 학습 | False |
| inference | False |
| thin upload | False |
| checkpoint load | False |
| torch import | False |
| 프론트 수정 | False |
| EODHD API call | False |
| EODHD fallback | False |

## 11. 실행 명령

```powershell
python scripts\cp115_yfinance_completed_day_append.py
```
