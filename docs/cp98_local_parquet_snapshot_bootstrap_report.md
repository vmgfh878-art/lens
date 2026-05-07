# CP98-D: local parquet snapshot bootstrap

작성일: 2026-05-04  
판정: PASS  
범위: CP95 100티커, yfinance, 1D, 2015-01-01 이후  
목표: CP97 재시도에 필요한 local parquet snapshot 3종 생성

## 1. Executive Summary

CP98은 PASS다.

CP97에서 막혔던 필수 local parquet snapshot 3개를 생성했다.

| 파일 | 결과 |
|---|---|
| `data/parquet/stock_info.parquet` | 생성 완료 |
| `data/parquet/price_data_yfinance.parquet` | 생성 완료 |
| `data/parquet/indicators_yfinance_1D.parquet` | 생성 완료 |

Supabase에서는 작은 `stock_info`만 읽었다. `price_data`와 `indicators`는 Supabase에서 읽거나 쓰지 않았다. 가격은 yfinance에서 직접 받고, indicator는 로컬 `price_data_yfinance.parquet`에서 계산했다.

핵심 검증 결과:

| 항목 | 결과 |
|---|---:|
| stock_info rows | 100 |
| yfinance price rows | 284,900 |
| yfinance price tickers | 100 |
| price date range | 2015-01-02 ~ 2026-05-01 |
| indicator rows | 279,000 |
| indicator tickers | 100 |
| indicator date range | 2015-03-30 ~ 2026-05-01 |
| duplicate `(ticker,date,source)` | 0 |
| duplicate `(ticker,timeframe,date,source)` | 0 |
| adjusted OHLC contract violation | 0 |
| indicator feature non-finite | 0 |
| ATR ratio coverage | 100% |
| source_data_hash | `3e4ee198` |
| local split gate | PASS |

## 2. 생성 방식

재현용 bootstrap 스크립트를 추가했다.

```text
scripts/cp98_local_parquet_snapshot_bootstrap.py
```

실행 환경:

```text
LENS_DATA_BACKEND=local
LENS_REQUIRE_LOCAL_SNAPSHOTS=1
LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\data\parquet
MARKET_DATA_PROVIDER=yfinance
```

실행 명령:

```text
python scripts\cp98_local_parquet_snapshot_bootstrap.py --output-dir data\parquet --limit-tickers 100 --start-date 2015-01-01 --metrics-path docs\cp98_local_parquet_snapshot_bootstrap_metrics.json
```

## 3. Universe

대상은 CP95 최종 100티커를 사용했다.

필수 포함 티커:

```text
AAPL, MSFT, NVDA, TSLA, NFLX
```

`stock_info`는 Supabase에서 이 100티커만 읽어 parquet로 저장했다. 누락 ticker는 0개다.

## 4. stock_info snapshot

| 항목 | 값 |
|---|---:|
| rows | 100 |
| missing tickers | 0 |
| file size | 7,088 bytes |
| path | `data/parquet/stock_info.parquet` |

`stock_info`는 작은 테이블이므로 CP98 허용 범위 안에서 Supabase read를 수행했다.

## 5. price_data_yfinance snapshot

| 항목 | 값 |
|---|---:|
| rows | 284,900 |
| tickers | 100 |
| date_min | 2015-01-02 |
| date_max | 2026-05-01 |
| duplicate `(ticker,date,source)` | 0 |
| failed tickers | 0 |
| checksum | `5e4c8f4286ad1dc1` |
| file size | 10,807,983 bytes |

저장된 source/provenance:

| 필드 | 값 |
|---|---|
| `source` | `yfinance` |
| `provider` | `yfinance` |
| `provider_adjustment_policy` | `yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc` |
| `updated_at` | 생성 시각 기록 |

가격 계약:

| 검증 | 결과 |
|---|---|
| required OHLC/adjusted_close null | 0 |
| adjusted factor invalid | 0 |
| adjusted high/low violation | 0 |
| duplicate date | 0 |
| volume negative | 0 |
| open/high/low ratio p99 sanity | PASS |

## 6. indicators_yfinance_1D snapshot

`indicators_yfinance_1D.parquet`는 `price_data_yfinance.parquet`를 읽어 `build_features(..., timeframe="1D")`로 로컬 계산했다.

| 항목 | 값 |
|---|---:|
| rows | 279,000 |
| tickers | 100 |
| date_min | 2015-03-30 |
| date_max | 2026-05-01 |
| duplicate `(ticker,timeframe,date,source)` | 0 |
| feature non-finite | 0 |
| atr_ratio non-null | 279,000 |
| atr_ratio coverage | 100% |
| checksum | `22930f99c9b9f775` |
| file size | 30,431,231 bytes |

`atr_ratio`는 indicator snapshot에는 포함했지만 모델 feature에는 포함하지 않았다.

## 7. Feature Contract

| 항목 | 결과 |
|---|---|
| `MODEL_N_FEATURES` | 36 |
| `FEATURE_CONTRACT_VERSION` | `v3_adjusted_ohlc` |
| `atr_ratio in MODEL_FEATURE_COLUMNS` | false |
| band feature set | `price_volatility_volume` |
| band feature count | 11 |

band feature set:

```text
log_return, open_ratio, high_ratio, low_ratio, vol_change,
ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position
```

## 8. source_data_hash 및 local split gate

local snapshot 기준 `source_data_hash`는 다음이다.

```text
3e4ee198
```

5티커 CP97 대상 기준 local split gate:

| 항목 | 값 |
|---|---:|
| tickers | AAPL, MSFT, NVDA, TSLA, NFLX |
| train samples | 9,345 |
| val samples | 2,000 |
| test samples | 2,010 |
| total samples | 13,355 |
| feature non-finite | 0 |
| target non-finite | 0 |
| feature_version | `v3_adjusted_ohlc` |

판정: local mode에서 `prepare_dataset_splits`가 snapshot 기반으로 동작한다.

## 9. Supabase Guard

확인 결과:

| 항목 | 결과 |
|---|---|
| Supabase `stock_info` read | 허용 범위 내 수행 |
| Supabase `price_data` read | 없음 |
| Supabase `indicators` read | 없음 |
| Supabase `price_data` write | 없음 |
| Supabase `indicators` write | 없음 |
| local mode `price_data` bulk read guard | PASS, 차단됨 |
| local mode `indicators` bulk read guard | PASS, 차단됨 |

대량 read guard 확인:

```text
price_data GUARD_PASS
indicators GUARD_PASS
```

## 10. 금지 작업 확인

| 금지 항목 | 위반 여부 |
|---|---|
| Supabase `price_data/indicators` 대량 read | 없음 |
| 전체 price_data Supabase write | 없음 |
| indicators Supabase full recompute | 없음 |
| model training | 없음 |
| inference save | 없음 |
| product run 교체 | 없음 |
| EODHD row 삭제 | 없음 |
| 1W/1M 처리 | 없음 |

## 11. 검증

| 검증 | 결과 |
|---|---|
| snapshot 3종 생성 | PASS |
| metrics JSON parse | PASS |
| py_compile | PASS |
| local split gate | PASS |
| feature/target finite | PASS |
| pytest | 미실행 |

`pytest` 미실행 사유:

```text
현재 Python 환경에 pytest 모듈이 없음
```

## 12. 산출물

| 산출물 | 경로 |
|---|---|
| stock_info snapshot | `data/parquet/stock_info.parquet` |
| price snapshot | `data/parquet/price_data_yfinance.parquet` |
| indicator snapshot | `data/parquet/indicators_yfinance_1D.parquet` |
| metrics | `docs/cp98_local_parquet_snapshot_bootstrap_metrics.json` |
| log metrics | `logs/cp98_local_parquet_snapshot_bootstrap/cp98_snapshot_bootstrap_metrics.json` |
| bootstrap script | `scripts/cp98_local_parquet_snapshot_bootstrap.py` |

## 13. 다음 단계

CP97을 재시도할 수 있다.

다음 CP97 재시도에서는:

1. `LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\data\parquet`를 유지한다.
2. line checkpoint와 band checkpoint를 각각 로드한다.
3. 5티커 최신 asof prediction을 dry-run으로 먼저 만든다.
4. `predictions`와 `prediction_evaluations`에는 latest-only row만 저장한다.
5. composite 저장은 계속 금지한다.
