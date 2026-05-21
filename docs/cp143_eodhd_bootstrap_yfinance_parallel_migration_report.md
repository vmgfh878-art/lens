# CP143-D EODHD bootstrap + yfinance parallel 전환 보고서

## 1. 요약

판정은 **PASS**이다.

EODHD 소규모 bootstrap 리허설과 yfinance local overlap 비교가 통과했다. 500티커 bootstrap execution CP로 넘어갈 수 있다. 단, yfinance와는 병렬 dataset 분리가 필수다.

## 2. EODHD 사용 가능성

| 항목 | 값 |
|---|---|
| API key present | True |
| sample status | PASS |
| requested tickers | 12 |
| success tickers | 12 |
| row count | 19116 |
| source limit hit | False |

제한 리허설은 AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, AVGO, AMD, ADBE, BRK.B 12개로 수행했다. EODHD 직접 호출은 12개 모두 성공했고, 샘플 parquet는 `logs/cp143_eodhd_bootstrap_yfinance_parallel_migration/eodhd_price_rehearsal_sample.parquet`에 저장했다.

중요한 운영 주의점: 기존 provider 호출 경로는 로컬 환경의 `HTTP_PROXY`/`HTTPS_PROXY` 설정을 타면 `127.0.0.1:9` 연결 거부로 빈 frame을 반환할 수 있었다. 이번 리허설 스크립트는 EODHD 키/쿼터 판정을 오염시키지 않도록 제한 샘플 호출에서 환경 프록시를 무시했다. 다음 EODHD 500 bootstrap 실행 CP에서는 EODHD fetch session의 프록시 정책을 명시하거나 실행 환경에서 차단 프록시를 제거해야 한다.

## 3. 500티커 bootstrap 예상

| 항목 | 값 |
|---|---|
| universe count | 503 |
| expected price calls | 503 |
| estimated minutes | 9.6 |

## 4. yfinance overlap 비교

| 항목 | 값 |
|---|---|
| status | PASS |
| overlap tickers | 11 |
| overlap rows | 17506 |
| adjusted_close median rel diff | 4.4207020042268803e-08 |
| adjusted_close p99 rel diff | 7.277795319980194e-05 |
| adjusted_factor p99 rel diff | 0.9750000006461327 |

해석은 명확하다. EODHD와 yfinance의 `adjusted_close`는 overlap 구간에서 거의 일치하지만, raw `close`와 `adjusted_factor`는 split 반영 정책 차이로 크게 벌어진다. 따라서 EODHD raw OHLC 뒤에 yfinance row를 이어 붙이는 혼합 append는 금지해야 한다. 모델/feature에서는 각 provider별 adjusted OHLC 재구성 계약을 유지하고, provider가 바뀌면 parquet 파일명, manifest, source hash, feature cache path가 모두 달라져야 한다.

## 5. 선택한 운영 전략

추천안은 **B안: EODHD bootstrap dataset과 yfinance parallel dataset 분리**이다.

A안처럼 EODHD baseline 뒤에 yfinance daily row를 같은 parquet에 append하면 provider adjustment policy가 경계일에서 섞인다. CP29 계열에서 raw/adjusted 혼용이 실제 병목이었던 만큼, 이번에는 source/provider별 dataset을 끝까지 분리한다.

## 6. local dataset naming

- `price_data_eodhd_500.parquet`
- `indicators_eodhd_1D_500.parquet`
- `indicators_eodhd_1W_500.parquet`
- `context/eodhd_500/*.parquet`
- `price_data_yfinance_500.parquet`
- `indicators_yfinance_1D_500.parquet`
- `indicators_yfinance_1W_500.parquet`
- `context/yfinance_500/*.parquet`

## 7. 금지 작업 미발생 확인

DB write, Supabase raw data write, 모델 학습, inference 저장, 프론트 수정, EODHD 500 full run은 수행하지 않았다.

## 8. 다음 CP 실행 조건

1. EODHD 500 bootstrap execution CP는 전체 503개 ticker를 바로 한 번에 밀지 말고 25~50 ticker 단위 checkpoint 방식으로 실행한다.
2. 각 batch마다 row count, date range, adjusted OHLC violation, duplicate ticker/date/source, failed ticker 사유를 기록한다.
3. EODHD provider 경로는 환경 프록시를 명시적으로 다루어야 한다.
4. yfinance 429가 풀리더라도 EODHD parquet에 yfinance append를 하지 않는다. yfinance는 `price_data_yfinance_500.parquet` 계열에 별도로 쌓는다.
5. EODHD quota remaining은 코드 경로에서 확인하지 못했으므로, 실행 전 dashboard 또는 별도 quota endpoint로 사람이 확인해야 한다.
