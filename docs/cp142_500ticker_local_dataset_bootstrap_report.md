# CP142-D 500티커 local yfinance dataset bootstrap 보고서

## 1. 요약

판정은 **FAIL**이다.

Yahoo preflight가 통과하지 못해 500티커 대량 수집을 안전 중단했다. 현재 상태에서 yfinance bootstrap을 진행하면 429를 확대할 위험이 있다.

## 2. universe

- 입력 파일: `backend/data/universe/sp500.csv`
- 요청 ticker 수: 503
- yfinance symbol 변환 예시: `BRK.B -> BRK-B`

## 3. yfinance preflight

| 항목 | 값 |
|---|---|
| status | FAIL_429 |
| reason | Yahoo chart API preflight가 모든 티커에서 429 Too Many Requests를 반환했다. |
| any_429 | True |
| any_json_ok | False |

## 4. 생성 파일

| 파일 | exists | rows | tickers | date_min | date_max |
|---|---:|---:|---:|---|---|
| price 1D 500 | False | None | None | None | None |
| indicators 1D 500 | False | None | None | None | None |
| indicators 1W 500 | False | None | None | None | None |

## 5. 검증

- adjusted OHLC violation: None
- duplicate price: None
- 1D feature status: NOT_RUN
- 1W feature status: NOT_RUN
- 1D split gate: NOT_RUN
- 1W split gate: NOT_RUN

## 6. 실패 ticker

- 실패 ticker 수: 503
- 실패 CSV: `docs/cp142_500ticker_failed_tickers.csv`
- 대표 실패 사유: {'FAIL_429': 503}

## 7. 금지 작업 미발생 확인

DB write, Supabase price_data/indicators 대량 read/write, EODHD 호출, 모델 학습/inference, W&B, 프론트 수정은 수행하지 않았다.
