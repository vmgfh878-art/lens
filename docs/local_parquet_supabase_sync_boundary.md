# 로컬 Parquet + Supabase 동기화 경계

생성일: 2026-05-06

## 1. 운영 경계

Lens의 데이터 운영 경계는 다음과 같다.

| 단계 | 데이터 위치 | write 대상 | 금지 |
|---|---|---|---|
| yfinance 가격 수집 | yfinance API -> 로컬 Parquet | `data/parquet/price_data_yfinance.parquet` | EODHD fallback |
| 1D indicator 갱신 | 로컬 가격 Parquet | `data/parquet/indicators_yfinance_1D.parquet` | Supabase indicator 전체 재계산 |
| feature/cache 생성 | 로컬 Parquet | `ai/cache` | Supabase price/indicator 대량 읽기 |
| 제품 inference | local parquet + checkpoint | 메모리, 로그 | full training |
| 제품 얇은 업로드 | inference output | Supabase `predictions`, `prediction_evaluations` 최신값 전용 | composite/history 대량 저장 |
| 제품 API 조회 | Supabase 얇은 DB | 읽기 전용 | price/indicator 전체 scan |

## 2. 일일 운영 순서

1. 환경 고정

```powershell
$env:MARKET_DATA_PROVIDER="yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER=""
$env:LENS_DATA_BACKEND="local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS="1"
$env:LENS_LOCAL_SNAPSHOT_DIR="C:\Users\user\lens\data\parquet"
$env:WANDB_MODE="disabled"
```

2. yfinance 완료 거래일 조회

규칙:
- `row.date < current_date`만 append
- 현재 날짜 row는 부분 daily 후보로 제외
- fallback provider는 비워둔다
- EODHD API key는 없어도 동작해야 한다

3. 로컬 가격 추가

검증:
- duplicate `(ticker, date, source) = 0`
- source/provider 모두 `yfinance`
- adjusted OHLC violation 0
- appended row count가 기대값과 일치

4. 1D indicator 증분 갱신

검증:
- append된 ticker만 rebuild
- duplicate `(ticker, timeframe, date, source) = 0`
- `atr_ratio` coverage 유지
- feature NaN/Inf 0
- 로컬 가격 최신일과 indicator 최신일 일치

5. 제품 inference

검증:
- line run: `patchtst-1D-efad3c29d803`
- band run: `cnn_lstm-1D-d0c780dee5e8`
- `asof_date == 로컬 indicator 최신일`
- forecast_dates length 5
- line/band/lower/upper series length 5
- lower <= upper
- composite row 0

6. Supabase 얇은 업로드

검증:
- `save_product_latest_predictions()` 사용
- line/band 최신값 전용 row만 저장
- predictions/evaluations row 수가 제한 범위
- product run_id 유지
- Supabase price_data/indicators write 없음

7. API readiness

검증:
- AAPL line latest asof_date == expected asof_date
- AAPL band latest asof_date == expected asof_date
- meta.layer line/band 분리
- EODHD fallback 0

## 3. 중단 조건

아래 중 하나라도 발생하면 daily loop를 중단한다.

- yfinance fetch 실패
- EODHD fallback 발생
- current date partial row가 append 대상에 포함됨
- local price latest date != indicator latest date
- duplicate ticker/date/source 발생
- adjusted OHLC violation 발생
- feature/target NaN/Inf 발생
- checkpoint run_id 불일치
- prediction asof_date가 local indicator latest date와 다름
- composite row 저장 시도
- Supabase `price_data`/`indicators` bulk read/write 시도
- latest-only row 제한 초과

## 4. 준비도와 최신성 체크

필수 체크:

| 체크 | 기대값 |
|---|---|
| 로컬 1D 가격 최신일 | 완료 거래일 |
| 로컬 1D indicator 최신일 | 로컬 가격 최신일과 동일 |
| source/provider | yfinance |
| source_data_hash | local snapshot 변경 시 변경 |
| 제품 line asof_date | 로컬 indicator 최신일과 동일 |
| 제품 band asof_date | 로컬 indicator 최신일과 동일 |
| Supabase 최신 asof_date | 제품 inference asof와 동일 |
| EODHD fallback | 0 |
| Supabase bulk read guard | PASS_BLOCKED |

## 5. Source/Hash 계약

source/hash mismatch가 나면 중단한다.

필수 포함:
- market_data_provider
- source
- provider_adjustment_policy
- feature_version
- timeframe
- ticker universe fingerprint
- date range
- indicator checksum 또는 Parquet hash

CP117 기준 현재 1D local snapshot:
- price latest: `2026-05-04`
- indicator latest: `2026-05-04`
- latest product asof: `2026-05-04`

## 6. 1W/1M 주의점

1W/1M은 아직 제품 loop closure 대상이 아니다.

1W/1M에서 추가 확인할 것:
- partial week/month 제외
- 1D parquet resample 기준 유지
- source/provider yfinance 유지
- 1W/1M 모델 후보 저장 전 latest-only 계약 재확인

## 7. 운영 명령 기준

현재 검증된 순서:

```powershell
python scripts\cp114_data_yfinance_freshness_check.py
python scripts\cp115_yfinance_completed_day_append.py
.\.venv\Scripts\python.exe scripts\cp116_eodhd_off_final_product_loop_rehearsal.py
```

CP117 이후에는 위 순서를 일일 로컬 운영으로 묶되, write 전 예행 점검/count gate를 유지한다.
