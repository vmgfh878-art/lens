# CP114-D yfinance 데이터 freshness 체크

생성일: 2026-05-05

## 1. Executive Summary

최종 판정: NEW_ROWS_AVAILABLE

yfinance가 snapshot 최신일보다 새로운 거래일 row를 제공했다. local parquet append와 1D indicator incremental refresh를 별도 CP로 진행할 수 있다. 단, end_date와 같은 날짜의 row가 포함되어 있어 장중 partial daily row 가능성을 append CP에서 걸러야 한다.

이번 체크는 CP103 스크립트를 사용하지 않았다. 모델, inference, checkpoint, torch import 없이 local parquet 최신일과 yfinance 5티커 최신 row만 확인했다.

## 2. Local Snapshot 최신일

| snapshot | rows | tickers | date min | date max | duplicate | source | provider |
|---|---:|---:|---|---|---:|---|---|
| 1D price | 284900 | 100 | 2015-01-02 | 2026-05-01 | 0 | ['yfinance'] | ['yfinance'] |
| 1D indicators | 279000 | 100 | 2015-03-30 | 2026-05-01 | 0 | ['yfinance'] | ['yfinance'] |
| 1W price | 59200 | 100 | 2015-01-02 | 2026-05-01 | 0 | ['yfinance'] | ['yfinance'] |
| 1W indicators | 53300 | 100 | 2016-02-19 | 2026-05-01 | 0 | ['yfinance'] | ['yfinance'] |

## 3. yfinance 최신 조회

요청:
- tickers: AAPL, MSFT, NVDA, TSLA, NFLX
- start_date: 2026-04-21
- end_date: 2026-05-05
- snapshot_latest_date: 2026-05-01
- fallback_used_any: False
- torch_imported: False
- potential_partial_current_day_rows: 5

| ticker | status | fetched rows | fetched max | new rows | new max | fallback | adjusted OHLC |
|---|---:|---:|---|---:|---|---:|---:|
| AAPL | PASS | 11 | 2026-05-05 | 2 | 2026-05-05 | False | True |
| MSFT | PASS | 11 | 2026-05-05 | 2 | 2026-05-05 | False | True |
| NFLX | PASS | 11 | 2026-05-05 | 2 | 2026-05-05 | False | True |
| NVDA | PASS | 11 | 2026-05-05 | 2 | 2026-05-05 | False | True |
| TSLA | PASS | 11 | 2026-05-05 | 2 | 2026-05-05 | False | True |

## 4. 판정

| 항목 | 값 |
|---|---|
| total_new_rows_after_snapshot_latest | 10 |
| all_adjusted_ohlc_passed | True |
| fallback_used_any | False |
| fetch_failed_count | 0 |
| new_dates_all | ['2026-05-04', '2026-05-05'] |
| partial_day_warning | end_date와 같은 신규 row가 있어 미국장 마감 전 partial daily row일 수 있다. append CP에서는 완료 거래일만 반영해야 한다. |
| 1D/1W 모델 실험 계속 가능 여부 | 조건부 가능: 기존 snapshot 기준 실험은 가능하지만, 최신 거래일 반영은 append/refresh CP 이후 권장한다. |
| EODHD 해지 gate 상태 | append/refresh 검증 전까지 보류 |

## 5. 다음 순서

사용자가 지정한 다음 순서를 유지한다.

1. 데이터 freshness 체크 결과 받기
2. CP114-BM/LM 결과 기반 1W 후보 정리
3. 1W 저장 후보 재현 CP
4. 그 다음 1D 추가 개선 또는 1W 제품 표시

## 6. 금지 항목 확인

| 항목 | 발생 |
|---|---:|
| 모델 학습 | False |
| inference 실행 | False |
| DB write | False |
| Supabase price_data/indicators 대량 read | False |
| thin upload | False |
| checkpoint load | False |
| torch import | False |
| 프론트 수정 | False |
| EODHD API call | False |
| EODHD fallback | False |

## 7. 실행 명령

```powershell
python scripts\cp114_data_yfinance_freshness_check.py
```
