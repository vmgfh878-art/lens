# CP-SUPA Supabase egress 사고 대응 보고서

작성일: 2026-05-04 KST  
상태: PASS_WITH_GUARDS  
핵심 판단: 2026-05-06 Free egress 제한 위험이 있으므로, `price_data`/`indicators` 대량 read/write는 즉시 중지하고 로컬 parquet snapshot을 기본 검증 경로로 전환한다.

## 1. Executive Summary

- Supabase egress의 1순위 원인은 Database API 대량 조회로 판단한다. `price_data`와 `indicators`를 feature cache, yfinance 비교, readiness, parquet export, 프론트 차트에서 반복 조회하는 경로가 있었다.
- Storage egress와 Realtime egress는 코드 검색 기준 Lens 앱에서 직접 사용하는 경로를 찾지 못했다. 단, Supabase 대시보드의 Usage 세부 그래프로 최종 확인이 필요하다.
- 이번 CP에서 `LENS_DATA_BACKEND=local` 또는 `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`일 때 `price_data`/`indicators` Supabase REST 대량 조회를 차단했다.
- feature index/source hash/feature cache 생성은 로컬 parquet snapshot을 우선 사용한다. provider/source/hash가 로컬 snapshot 기준으로 계산되어 Supabase 반복 read를 피한다.
- 프론트 Stock/Backtest 기본 가격 조회는 2015년 이후 전체 조회에서 최근 365일 단일 요청으로 축소했다. 2015년 전체 history는 명시 호출(`fullHistory=true`)일 때만 실행되는 lazy 경로로 남겼다.

## 2. 즉시 중지 절차

### Render cron

1. Render Dashboard로 이동한다.
2. Cron Jobs에서 `lens-daily-market-sync`를 선택한다.
3. Suspend/Pause를 선택한다.
4. `render.yaml` 기준 현재 cron은 `python -m backend.collector.pipelines.daily_market_sync --indicator-lookback-days 60`이다. 이 job은 가격 sync와 indicator compute를 호출하므로 egress 사고 중 실행 금지다.

### 로컬 자동 실행

- 아래 스크립트는 egress 사고 중 쓰기 모드로 실행하지 않는다.
- `scripts/run_daily_local_market_sync.ps1`
- `scripts/sync_yfinance_prices.ps1 -Write`
- `python -m backend.collector.pipelines.daily_market_sync`
- `python -m backend.collector.pipelines.daily_sync`
- `python -m backend.collector.pipelines.bootstrap_backfill`
- `python -m backend.collector.jobs.compute_indicators`
- `python -m backend.collector.pipelines.compute_indicators_cli`

Windows 예약 작업을 쓰고 있다면:

```powershell
Get-ScheduledTask | Where-Object { $_.TaskName -like "*Lens*" }
Disable-ScheduledTask -TaskName "<작업명>"
```

### 응급 환경변수

```powershell
$env:LENS_DATA_BACKEND = "local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS = "1"
$env:LENS_LOCAL_SNAPSHOT_DIR = "C:\Users\user\lens\data\parquet"
$env:LENS_READINESS_DB_ROW_LIMIT = "500"
```

이 모드에서는 로컬 parquet가 없으면 `price_data`/`indicators` 대량 read 대신 명확히 실패한다.

## 3. Supabase egress 원인 조사

| 구분 | 현재 판단 | 근거 | 조치 |
|---|---:|---|---|
| Database API egress | 높음 | `fetch_all_rows`/`fetch_frame`가 `price_data`, `indicators`를 page 단위로 읽고, feature cache/yfinance 검증/readiness/export/frontend가 반복 호출 가능 | 로컬 snapshot 우선, local mode 대량 REST 차단 |
| Storage egress | 낮음 추정 | 앱 코드에서 Supabase Storage bucket/download/signed URL 직접 사용 경로를 찾지 못함 | Supabase Dashboard Usage에서 최종 확인 |
| Realtime egress | 낮음 추정 | 앱 코드에서 Supabase realtime channel 사용 경로를 찾지 못함 | Realtime 기능 비활성 상태 확인 |

주의: 이번 조사는 코드 기반 감사다. Supabase 대시보드에서 Database API, Storage, Realtime egress 그래프를 각각 확인해야 최종 확정된다.

## 4. 구현 변경

### 로컬 snapshot repository

- 신규: `backend/collector/repositories/local_snapshots.py`
- 기본 경로: `data/parquet`
- 환경변수:
  - `LENS_LOCAL_SNAPSHOT_DIR`
  - `LENS_DATA_BACKEND=local|parquet|snapshot`
  - `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`
  - `LENS_USE_LOCAL_SNAPSHOTS=0`이면 자동 사용 해제
- 지원 파일명 예:
  - `price_data_yfinance.parquet`
  - `price_data_eodhd.parquet`
  - `indicators_yfinance_1D.parquet`
  - `indicators_eodhd_1D.parquet`
  - `stock_info.parquet`

### Supabase REST 대량 read guard

- 변경: `backend/collector/repositories/base.py`
- `LENS_DATA_BACKEND=local` 또는 `LENS_REQUIRE_LOCAL_SNAPSHOTS=1` 상태에서 `price_data`/`indicators`를 `limit` 없이 REST 조회하면 실패한다.
- 목적: feature build, diagnostics, readiness가 실수로 Supabase 전체 history를 다시 읽는 상황 차단.

### feature cache/source hash 로컬 전환

- 변경: `ai/preprocessing.py`
- `resolve_data_fingerprint()`가 로컬 `price_data`/`indicators` parquet에서 provider/source/date/count/checksum을 계산할 수 있게 했다.
- `fetch_feature_index_frame()`와 `fetch_training_frames()`는 로컬 snapshot을 우선 사용한다.
- local mode에서 snapshot이 없으면 DB fallback 없이 실패한다.
- indicator value checksum은 로컬 frame 값 기준으로 계산되어, 로컬 indicators 값이 바뀌면 cache hash도 바뀐다.

### yfinance 검증 local file first

- 변경: `backend/collector/pipelines/yfinance_price_sync.py`
- EODHD baseline 비교는 먼저 `price_data_eodhd.parquet`를 읽는다.
- local mode에서 parquet가 없으면 Supabase baseline read를 실행하지 않고 실패한다.
- yfinance write는 이번 CP에서 실행하지 않았다.

### readiness 대량 조회 금지

- 변경: `backend/collector/readiness.py`
- readiness는 로컬 snapshot을 먼저 사용한다.
- DB fallback은 `LENS_READINESS_DB_ROW_LIMIT`로 제한한다.
- local mode에서는 snapshot 없을 때 DB 조회 없이 실패한다.

### API read local snapshot 우선

- 변경: `backend/app/repositories/market_repo.py`
- 가격 API, 보조지표 API, 종목 검색 fallback은 local snapshot을 먼저 읽는다.
- local mode에서 snapshot이 없으면 Supabase 조회 없이 실패한다.
- 프론트 기본 요청량 축소와 별도로, API 저장소 레벨에서도 DB API 우회를 제공한다.

### parquet export 안전장치

- 변경: `backend/db/scripts/export_parquet.py`
- `price_data`/`indicators` export는 `--confirm-egress-export` 없이는 실행되지 않는다.
- 이유: snapshot 생성 자체도 Supabase egress를 쓰므로 사고 중 무심코 실행하면 제한 위험을 키운다.

### 프론트 기본 조회량 축소

- 변경: `frontend/src/components/StockView.tsx`
- 변경: `frontend/src/components/BacktestView.tsx`
- 기존: 2015년부터 현재까지 연도별 병렬 가격 API 호출.
- 변경: 기본 가격 조회는 최근 365일 단일 요청.
- `StockView`에는 전체 history 명시 호출용 `fullHistory=true` 경로만 남겼다.
- indicator 기본 요청량은 1000에서 300으로 축소했다.

## 5. 로컬 parquet 운영 계약

권장 snapshot 파일:

```text
data/parquet/
  stock_info.parquet
  price_data_eodhd.parquet
  price_data_yfinance.parquet
  indicators_eodhd_1D.parquet
  indicators_yfinance_1D.parquet
  macroeconomic_indicators.parquet
```

응급 기간에는 Supabase에서 새로 대량 export하지 말고, 이미 존재하는 로컬 snapshot 또는 yfinance local file first 검증을 우선 사용한다. 정말 snapshot을 새로 떠야 한다면 제한 ticker/date 범위로 별도 스크립트를 만들고, Supabase egress 잔량 확인 후 1회만 실행한다.

## 6. 남은 리스크

| 등급 | 리스크 | 상태 | 다음 조치 |
|---|---|---|---|
| P0 | Render `lens-daily-market-sync`가 계속 돌면 DB read/write와 indicator compute가 재발 | 운영 조치 필요 | Render cron Suspend |
| P0 | `compute_indicators` full recompute 실행 시 price_data 대량 read/write 발생 | 코드 guard 일부, 운영 금지 필요 | 사고 기간 실행 금지 |
| P1 | 기존 diagnostic/CP 스크립트 일부가 여전히 `fetch_frame("price_data")` 직접 사용 | local mode guard로 REST 대량 read 차단 | 필요한 스크립트만 local snapshot 대응 |
| P1 | snapshot 생성 자체가 Supabase egress를 사용할 수 있음 | export guard 추가 | egress 잔량 확인 전 export 금지 |
| P2 | API 서버의 가격 endpoint는 여전히 요청 범위가 크면 DB를 읽음 | 프론트 기본 호출 축소 | API limit/date range hard cap 별도 CP |
| P2 | Storage/Realtime egress는 코드상 낮아 보이나 대시보드 확인 전 확정 불가 | 추정 | Supabase Usage 확인 |

## 7. 검증 결과

실행한 명령:

```powershell
python -m py_compile backend\app\repositories\market_repo.py ai\preprocessing.py backend\collector\repositories\local_snapshots.py backend\collector\repositories\base.py backend\collector\pipelines\yfinance_price_sync.py backend\collector\readiness.py backend\db\scripts\export_parquet.py
$env:PYTHONPATH='C:\Users\user\lens;C:\Users\user\lens\backend'; python -m unittest ai.tests.test_preprocessing_cache_isolation backend.tests.test_market_data_providers backend.tests.test_services
.\node_modules\.bin\tsc --noEmit
```

결과:

- `py_compile` PASS
- `unittest` PASS, 30 tests
- `tsc --noEmit` PASS
- `pytest`는 현재 환경에 설치되어 있지 않아 실행하지 못했다.

## 8. 금지 준수 확인

- 전체 yfinance write 실행 안 함.
- indicators full recompute 실행 안 함.
- full model training 실행 안 함.
- live inference 실행 안 함.
- Supabase 대량 read 실행 안 함.
- DB write 실행 안 함.

## 9. 다음 조치

1. Render `lens-daily-market-sync` suspend 확인.
2. Supabase Dashboard에서 2026-05-04 기준 egress breakdown 확인: Database API, Storage, Realtime.
3. `data/parquet` snapshot 보유 현황 확인.
4. yfinance 검증은 `LENS_DATA_BACKEND=local`로 local snapshot 기반만 실행.
5. 별도 CP에서 API endpoint 단위 date range/limit hard cap 추가.
