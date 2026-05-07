# CP144-D EODHD 500티커 local bootstrap execution 보고서

## 1. 요약

판정은 **PASS**이다.

EODHD 500티커 local price bootstrap이 완료되어 다음 indicator/context generation CP로 진행 가능하다.

## 2. 실행 범위

| 항목 | 값 |
|---|---|
| universe | `backend\data\universe\sp500.csv` |
| universe ticker count | 503 |
| start_date | 2015-01-01 |
| end_date | 2026-05-06 |
| batch_size | 40 |
| provider/source | eodhd/eodhd |
| adjustment_policy | eodhd_raw_ohlc_adjusted_close_factor_v3_adjusted_ohlc |

## 3. 수집 결과

| 항목 | 값 |
|---|---|
| fetched ticker count | 503 |
| failed_empty | 0 |
| failed_contract | 0 |
| failed_http | 0 |
| failed_proxy | 0 |
| retry_pending | 0 |
| source_limit_hit | False |

## 4. 데이터 품질 검증

| 항목 | 값 |
|---|---|
| row count | 1387834 |
| ticker count | 503 |
| date min | 2015-01-02 |
| date max | 2026-05-05 |
| duplicate ticker/date/source | 0 |
| adjusted OHLC violation | 0 |
| missing OHLC count | 0 |
| volume null count | 0 |
| volume negative count | 0 |
| yfinance row count | 0 |

## 5. 저장 파일

- `data/parquet/price_data_eodhd_500.parquet`
- `data/parquet/price_data_eodhd_500.manifest.json`
- `logs/cp144_eodhd_500_bootstrap/ticker_status.csv`
- `docs/cp144_eodhd_500ticker_failed_tickers.csv`

## 6. 프록시 정책

이번 bootstrap은 EODHD session에서 환경 프록시를 사용하지 않는 정책으로 실행했다. 로컬 `HTTP_PROXY`/`HTTPS_PROXY`가 차단 프록시로 잡히면 정상 API key도 빈 frame처럼 보일 수 있기 때문이다. metrics와 manifest에 `use_env_proxy=false`로 기록했다.

## 7. source/provider 분리

생성 parquet의 `source`, `provider`는 모두 `eodhd`이고 yfinance row는 0개다. yfinance 병렬 dataset은 `price_data_yfinance_500.parquet` 계열로 별도 생성해야 하며, EODHD parquet에 yfinance daily row를 append하지 않는다.

## 8. 실패 티커

실패 또는 retry 대상은 `docs\cp144_eodhd_500ticker_failed_tickers.csv`에 기록했다. 실패가 0개면 다음 indicator/context generation CP로 바로 진행 가능하다.

## 9. 금지 작업 미발생 확인

Supabase raw write, DB write, yfinance append, 모델 학습, inference 저장, 프론트 수정, W&B 실행은 수행하지 않았다.
