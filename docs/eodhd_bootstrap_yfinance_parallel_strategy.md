# EODHD bootstrap + yfinance parallel 전략

## 결론

추천안은 **B안: EODHD bootstrap dataset과 yfinance parallel dataset 분리**이다.

EODHD로 500티커 초기 history를 빠르게 확보하되, yfinance daily append를 같은 parquet series에 섞지 않는다. 두 provider는 adjusted close, split/dividend 반영, ticker symbol, 누락일 정책이 다를 수 있으므로 source/provider별 파일과 manifest를 분리한다.

## 파일명 계약

- `data/parquet/price_data_eodhd_500.parquet`
- `data/parquet/indicators_eodhd_1D_500.parquet`
- `data/parquet/indicators_eodhd_1W_500.parquet`
- `data/parquet/context/eodhd_500/*.parquet`
- `data/parquet/price_data_yfinance_500.parquet`
- `data/parquet/indicators_yfinance_1D_500.parquet`
- `data/parquet/indicators_yfinance_1W_500.parquet`
- `data/parquet/context/yfinance_500/*.parquet`

## 운영 원칙

1. EODHD dataset은 bootstrap baseline이다.
2. yfinance dataset은 별도 parallel dataset으로 천천히 쌓는다.
3. 같은 ticker/date라도 provider가 다르면 같은 parquet에 append하지 않는다.
4. 모델 학습/feature cache는 `market_data_provider`, `provider_adjustment_policy`, `source_data_hash`, context checksum을 포함한다.
5. Supabase에는 raw price/indicator를 올리지 않고 product latest/thin 결과만 저장한다.
6. EODHD 실행 환경은 `HTTP_PROXY`/`HTTPS_PROXY`가 차단 프록시로 잡히지 않았는지 확인한다. 이번 CP143 제한 리허설에서는 환경 프록시를 무시한 직접 세션에서 EODHD 응답이 정상임을 확인했다.

## 금지할 혼합 방식

- `price_data_eodhd_500.parquet` 끝에 yfinance 신규 row append
- EODHD indicators와 yfinance price를 조합해 feature 생성
- provider가 다른 feature cache path 재사용
- provider 차이를 무시한 500티커 model run 비교

## 다음 CP 순서

1. EODHD 500 bootstrap execution CP
2. EODHD 500 indicators/context generation CP
3. 500 model training/smoke CP
4. yfinance parallel collector CP
5. EODHD baseline과 yfinance parallel reconciliation CP
