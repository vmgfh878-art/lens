# CP151-DG yfinance 500 backfill 보고서

## 1. 요약

최종 상태: **DRY_RUN_FETCH_BLOCKED**

이 CP는 EODHD 500 parquet를 수정하지 않고, 별도 yfinance 500 parquet 경로를 생성/갱신하는 작업이다. 기존 100 ticker 운영 parquet도 덮어쓰지 않았다.

## 2. 실행 설정

| 항목 | 값 |
|---|---|
| apply | False |
| start_date | 2015-01-01 |
| end_date | 2026-05-08 |
| chunk_size | 1 |
| max_chunks_per_run | 0 |
| selected_ticker_count | 0 |
| universe_ticker_count | 503 |

## 3. Price 결과

| 항목 | 값 |
|---|---:|
| success_ticker_count | 0 |
| failed_ticker_count | 0 |
| incoming_rows | 0 |
| appended_rows | 0 |
| skipped_existing_rows | 0 |
| price_row_count | 1383438 |
| price_ticker_count | 501 |
| price_date_max | 2026-05-08 |
| fallback_used_count | 0 |

## 4. Indicator 결과

| timeframe | rows | duplicate | nonfinite | mode | affected_ticker_count | status |
|---|---:|---:|---:|---|---:|---|
| 1D | None | None | None | None | None | None |
| 1W | None | None | None | None | None | None |

## 5. Latest Distribution

- latest distribution CSV: `docs\daily_yfinance_500_context_runs\20260508_20260516_200800\price_append_latest_distribution.csv`
- complete_to_end_date ratio: 0.9960238568588469
- complete_ticker_count: 501
- date_max: 2026-05-08

## 6. 실패 티커

- failed ticker CSV: `docs\daily_yfinance_500_context_runs\20260508_20260516_200800\price_append_failed_tickers.csv`
- 실패 티커는 state에 남기고 다음 run에서 retry한다.

## 7. 자동화 resume/idempotent 확인

- automation status: not_run
- automation apply: not_run
- selected_ticker_count: not_run
- appended_rows: not_run
- note: not_run

## 8. 금지 작업 확인

- EODHD fallback: False
- Supabase bulk read/write: False
- DB write: False
- model training: False
- inference save: False
- frontend modify: False

## 9. 예상 소요 시간

- best: direct chart fetch가 안정적이면 당일 완료.
- base: chunk 50 기준 10개 chunk와 indicator full build를 포함해 1~2일.
- worst: Yahoo 429, empty response, contract retry가 반복되면 3~5일.

## 10. Indicator 운영 모드

- 최초 500 백필은 full indicator build를 허용한다.
- 기존 indicator parquet가 존재하고 신규 append ticker만 생기는 이후 run은 `--indicator-mode auto`에서 incremental refresh로 전환한다.
- 강제 전체 재계산이 필요할 때만 `--indicator-mode full`을 명시한다.

## 11. 판단

dry-run이므로 parquet/state를 쓰지 않았다.

이 CP는 EODHD 해지 확정 CP가 아니다. 목적은 EODHD 500을 대체할 수 있는 yfinance 500 local dataset을 실제로 만드는 것이다.

해지 판단은 yfinance 500 parquet와 indicator가 2~3회 자동화에서 안정적으로 갱신되는지 확인한 뒤 해야 한다.
