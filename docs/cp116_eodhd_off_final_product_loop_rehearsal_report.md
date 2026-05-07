# CP116-D EODHD-off 1D 제품 루프 최종 리허설

생성일: 2026-05-05

## 1. Executive Summary

최종 판정: PASS

EODHD 없이 최신 asof_date=2026-05-04 기준 1D 제품 line/band inference, latest-only thin upload, API 조회가 모두 통과했다.

이번 CP는 CP115에서 local yfinance snapshot이 완료 거래일 `2026-05-04`까지 올라간 뒤, EODHD 없이 1D 제품 line/band checkpoint inference, latest-only thin upload, API latest 조회가 가능한지 최종 확인했다.

## 2. 환경 및 Snapshot

| 항목 | 값 |
|---|---|
| MARKET_DATA_PROVIDER | yfinance |
| MARKET_DATA_FALLBACK_PROVIDER |  |
| settings provider | yfinance |
| settings fallback | None |
| EODHD key present | False |
| LENS_DATA_BACKEND | local |
| WANDB_MODE | disabled |

| snapshot | rows | latest date | duplicate | source | provider |
|---|---:|---|---:|---|---|
| 1D price | 284905 | 2026-05-04 | 0 | ['yfinance'] | ['yfinance'] |
| 1D indicators | 279005 | 2026-05-04 | 0 | ['yfinance'] | ['yfinance'] |

## 3. Bulk Read Guard

| table | result |
|---|---|
| price_data | PASS_BLOCKED |
| indicators | PASS_BLOCKED |

## 4. 제품 Inference

| 항목 | line | band |
|---|---:|---:|
| run_id | patchtst-1D-efad3c29d803 | cnn_lstm-1D-d0c780dee5e8 |
| model | patchtst | cnn_lstm |
| asof_dates | ['2026-05-04'] | ['2026-05-04'] |
| prediction_count | 5 | 5 |
| evaluation_count | 5 | 5 |
| forecast_date_lengths | [5] | [5] |
| lower_lte_upper | True | True |

전체:
- prediction_rows: 10
- evaluation_rows: 10
- all_forecast_horizon_5: True
- all_series_length_5: True
- lower_lte_upper: True
- composite_rows: 0

## 5. Latest-only Thin Upload

| 항목 | 값 |
|---|---:|
| attempted | True |
| prediction_rows_written | 10 |
| evaluation_rows_written | 10 |
| before_counts | {'line_predictions': 1870, 'band_predictions': 2015, 'line_evaluations': 1870, 'band_evaluations': 2015} |
| after_counts | {'line_predictions': 1875, 'band_predictions': 2020, 'line_evaluations': 1875, 'band_evaluations': 2020} |
| count_deltas | {'line_predictions': 5, 'band_predictions': 5, 'line_evaluations': 5, 'band_evaluations': 5} |

## 6. API Latest 조회

| 항목 | 값 |
|---|---|
| AAPL line found | True |
| AAPL band found | True |
| line asof_date | 2026-05-04 |
| band asof_date | 2026-05-04 |
| expected asof_date | 2026-05-04 |
| asof_date matched | True |
| line meta layer | line |
| band meta layer | band |

## 7. EODHD 해지 가능 최종 판정

EODHD 해지 가능 판정. yfinance/local parquet append/indicator refresh와 최신 1D 제품 loop가 모두 통과했다.

## 8. 금지 작업 미발생 확인

| 항목 | 발생 |
|---|---:|
| Supabase price_data/indicators 대량 read/write | False |
| 전체 universe write | False |
| full/model retraining | False |
| composite 저장 | False |
| EODHD API call | False |
| EODHD fallback | False |
| 프론트 수정 | False |
| DB row delete | False |

## 9. 남은 운영 주의점

- yfinance 최신 row 중 현재 날짜 row는 partial일 수 있으므로 CP115의 `row.date < current_date` 필터를 daily 운영에도 유지한다.
- Supabase에는 price/indicator 원본을 올리지 않고 product latest prediction만 얇게 저장한다.
- EODHD 해지 후에도 provider 장애 시 재시도/지연 운영 절차가 필요하다.
- 1W/1M 제품화 전에는 별도 snapshot append/partial period guard를 다시 확인한다.

## 10. 실행 명령

```powershell
.\.venv\Scripts\python.exe scripts\cp116_eodhd_off_final_product_loop_rehearsal.py
```
