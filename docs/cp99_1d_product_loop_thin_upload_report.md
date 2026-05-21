# CP99-D: 1D 제품 루프 thin upload 검증

작성일: 2026-05-04  
판정: PASS  
범위: 1D, horizon 5, provider/source yfinance, 5티커 latest-only 저장  
대상 run:

| layer | run_id |
|---|---|
| line | `patchtst-1D-efad3c29d803` |
| band | `cnn_lstm-1D-d0c780dee5e8` |

## 1. Executive Summary

CP99는 PASS다.

CP98에서 생성한 local yfinance parquet snapshot만 사용해 1D 제품 line/band 예측을 만들었고, Supabase에는 제품 화면에 필요한 5티커 latest-only 결과만 얇게 저장했다.

핵심 결과:

| 항목 | 결과 |
|---|---:|
| snapshot source_data_hash | `3e4ee198` |
| 대상 ticker | AAPL, MSFT, NVDA, TSLA, NFLX |
| line prediction rows | 5 |
| band prediction rows | 5 |
| prediction_evaluations rows | 10 |
| latest asof_date | 2026-05-01 |
| forecast horizon | 5 |
| line/conservative/lower/upper length | 모두 5 |
| lower <= upper | PASS |
| Supabase price_data/indicators 대량 read | 없음 |
| composite 저장 | 없음 |
| API repository 조회 | PASS |

주의: 최신 asof_date는 오늘 날짜가 아니라 local snapshot의 최신 거래일인 `2026-05-01` 기준이다.

## 2. Local Snapshot Gate

사용 환경:

```text
LENS_DATA_BACKEND=local
LENS_REQUIRE_LOCAL_SNAPSHOTS=1
LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\data\parquet
MARKET_DATA_PROVIDER=yfinance
```

snapshot 상태:

| snapshot | rows | tickers | date range |
|---|---:|---:|---|
| `stock_info.parquet` | 100 | 100 | 해당 없음 |
| `price_data_yfinance.parquet` | 284,900 | 100 | 2015-01-02 ~ 2026-05-01 |
| `indicators_yfinance_1D.parquet` | 279,000 | 100 | 2015-03-30 ~ 2026-05-01 |

계약 확인:

| 항목 | 결과 |
|---|---|
| price source values | `yfinance` |
| indicator source values | `yfinance` |
| duplicate `(ticker,date,source)` | 0 |
| duplicate `(ticker,timeframe,date,source)` | 0 |
| `MODEL_N_FEATURES` | 36 |
| feature version | `v3_adjusted_ohlc` |
| `atr_ratio` 모델 feature 포함 | false |
| source_data_hash | `3e4ee198` |

local split/finite gate:

| 항목 | 값 |
|---|---:|
| train samples | 9,345 |
| val samples | 2,000 |
| test samples | 2,010 |
| feature non-finite | 0 |
| target non-finite | 0 |

## 3. Inference Dry-Run

기존 checkpoint만 로드했다. 새 학습은 실행하지 않았다.

| layer | model | seq_len | feature columns | predictions | evaluations | asof_date |
|---|---|---:|---:|---:|---:|---|
| line | patchtst | 252 | 36 | 5 | 5 | 2026-05-01 |
| band | cnn_lstm | 60 | 11 | 5 | 5 | 2026-05-01 |

형상 검증:

| 검증 | 결과 |
|---|---|
| forecast_dates length | 5 |
| line_series length | 5 |
| conservative_series length | 5 |
| lower_band_series length | 5 |
| upper_band_series length | 5 |
| lower <= upper | PASS |

dry-run payload:

| 항목 | bytes |
|---|---:|
| line prediction payload | 6,277 |
| band prediction payload | 6,305 |
| predictions + evaluations 전체 payload | 16,191 |

dry-run artifact:

```text
logs/cp99_1d_product_loop_thin_upload/dry_run_predictions.json
```

## 4. Thin Upload

저장은 latest-only로 제한했다. `ai.inference --save`는 split 전체 저장 위험이 있어 사용하지 않았다.

저장 대상:

| 테이블 | line | band | 합계 |
|---|---:|---:|---:|
| `predictions` | 5 | 5 | 10 |
| `prediction_evaluations` | 5 | 5 | 10 |

저장 전후 count:

| 항목 | before | after | 증가 |
|---|---:|---:|---:|
| line predictions | 1,865 | 1,870 | 5 |
| band predictions | 2,010 | 2,015 | 5 |
| line evaluations | 1,865 | 1,870 | 5 |
| band evaluations | 2,010 | 2,015 | 5 |

이는 과거 history 전체 저장이 아니라 5티커 latest-only 추가임을 확인한다.

## 5. API 확인

프론트 UI는 수정하지 않았다. 백엔드 repository 경로에서 AAPL 1D 최신 prediction을 run_id 기준으로 조회했다.

| 조회 | 결과 |
|---|---|
| AAPL line latest | PASS |
| AAPL band latest | PASS |
| line run_id | `patchtst-1D-efad3c29d803` |
| band run_id | `cnn_lstm-1D-d0c780dee5e8` |
| line meta layer | `line` |
| band meta layer | `band` |
| asof_date | 2026-05-01 |

1W/1M 정책은 이번 CP에서 건드리지 않았다.

## 6. Supabase Guard

| 항목 | 결과 |
|---|---|
| Supabase `price_data` 대량 read | 차단 확인 |
| Supabase `indicators` 대량 read | 차단 확인 |
| Supabase price_data write | 없음 |
| Supabase indicators recompute/write | 없음 |
| predictions/evaluations write | latest-only 20 rows |

대량 read guard:

```text
price_data: PASS_BLOCKED
indicators: PASS_BLOCKED
```

## 7. 금지 작업 확인

| 금지 항목 | 위반 여부 |
|---|---|
| 전체 yfinance Supabase write | 없음 |
| Supabase price_data/indicators 대량 read | 없음 |
| indicators Supabase recompute | 없음 |
| full training | 없음 |
| 1W/1M 처리 | 없음 |
| prediction history 대량 저장 | 없음 |
| EODHD 삭제 | 없음 |
| 프론트 UI 수정 | 없음 |
| composite 저장 | 없음 |

## 8. 검증

| 검증 | 결과 |
|---|---|
| py_compile | PASS |
| metrics JSON parse | PASS |
| API repository check | PASS |
| local bulk read guard | PASS |
| pytest | 미실행 |

`pytest` 미실행 사유:

```text
현재 Python 환경에 pytest 모듈이 없음
```

## 9. 산출물

| 산출물 | 경로 |
|---|---|
| metrics | `docs/cp99_1d_product_loop_thin_upload_metrics.json` |
| log metrics | `logs/cp99_1d_product_loop_thin_upload/cp99_1d_product_loop_thin_upload_metrics.json` |
| dry-run predictions | `logs/cp99_1d_product_loop_thin_upload/dry_run_predictions.json` |
| 실행 스크립트 | `scripts/cp99_1d_product_loop_thin_upload.py` |

## 10. 최종 판단

1D 제품 루프는 `local parquet + yfinance + Supabase thin DB` 구조로 닫을 수 있다.

닫힌 범위:

| 범위 | 상태 |
|---|---|
| 1D local yfinance snapshot 기반 inference | PASS |
| line/band 제품 run latest prediction | PASS |
| 5티커 latest-only thin upload | PASS |
| API repository 조회 | PASS |
| composite 미사용 | PASS |

아직 닫히지 않은 범위:

| 범위 | 이유 |
|---|---|
| 전체 universe latest upload | 별도 batch gate 필요 |
| 1W/1M | 이번 CP 금지 범위 |
| 프론트 실제 화면 육안 확인 | UI 수정 없이 repository 조회까지만 확인 |
