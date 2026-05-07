# Supabase 비용 가드 체크리스트

작성일: 2026-05-04 KST  
목적: egress/DB size가 Free 한도를 다시 넘지 않도록 실행 전 확인할 운영 체크리스트를 둔다.

## 1. 즉시 운영 상태

- [ ] Render `lens-daily-market-sync`가 Suspended/Paused 상태다.
- [ ] Supabase Usage에서 Database API, Storage, Realtime egress를 각각 확인했다.
- [ ] `LENS_DATA_BACKEND=local` 또는 `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`로 로컬 검증 모드를 켰다.
- [ ] `LENS_LOCAL_SNAPSHOT_DIR`가 실제 parquet snapshot 경로를 가리킨다.
- [ ] `price_data`/`indicators` 대량 read가 필요한 작업은 실행하지 않는다.

## 2. 실행 금지 명령

아래는 사용자 승인과 별도 CP 없이는 실행 금지다.

```powershell
python -m backend.collector.pipelines.daily_market_sync
python -m backend.collector.pipelines.daily_sync
python -m backend.collector.pipelines.bootstrap_backfill
python -m backend.collector.pipelines.compute_indicators_cli
python -m backend.db.scripts.export_parquet --tables price_data indicators --confirm-egress-export
.\scripts\sync_yfinance_prices.ps1 -Write
.\scripts\run_daily_local_market_sync.ps1
```

## 3. 대량 read 전 체크

- [ ] 이 작업이 Supabase 대신 local parquet를 읽는지 확인했다.
- [ ] `price_data`/`indicators` 전체 기간을 읽지 않는다.
- [ ] API 조회라면 start/end 또는 limit이 있다.
- [ ] readiness는 `LENS_READINESS_DB_ROW_LIMIT` 제한을 사용한다.
- [ ] export가 필요하면 제한 ticker/date 조건과 egress 잔량을 먼저 확인했다.

## 4. 대량 write/upsert 전 체크

- [ ] `chunked_upsert()` 기본값이 `returning=ReturnMethod.minimal`이다.
- [ ] 대량 write 명령에 `--write`가 있다면 ticker/date 범위가 제한되어 있다.
- [ ] 전체 yfinance write가 아니다.
- [ ] indicators full recompute가 아니다.
- [ ] 기존 EODHD row 삭제/복구를 같이 하지 않는다.
- [ ] write 전 parquet backup과 manifest가 있다.
- [ ] write 후 read 검증도 제한 범위로만 수행한다.

## 5. DB 슬리밍 전 체크

- [ ] 용량 추정 SQL은 `pg_total_relation_size`와 추정 row 기반이다.
- [ ] `price_data`/`indicators`에 `count(*)` 전체 스캔을 하지 않는다.
- [ ] 제품 run 보호 목록을 확정했다.
- [ ] 삭제 후보는 먼저 parquet export했다.
- [ ] export row_count/checksum 검증을 완료했다.
- [ ] 삭제 SQL은 별도 CP와 사용자 승인 후 실행한다.

## 6. Supabase에 남길 것

- [ ] `stock_info`
- [ ] 제품 표시용 최신 가격 일부
- [ ] 제품 표시용 최신 보조지표 일부
- [ ] 제품 line/band run의 `model_runs`
- [ ] 제품 line/band run의 최신 `predictions`
- [ ] 제품 run의 최신 `prediction_evaluations`
- [ ] 제품 데모용 최소 `backtest_results`

## 7. 로컬로 내릴 것

- [ ] 전체 `price_data`
- [ ] 전체 `indicators`
- [ ] 학습 feature/cache/index/ticker registry
- [ ] 실험용 `predictions`
- [ ] 실험용 `prediction_evaluations`
- [ ] legacy/composite/failed/smoke run archive
- [ ] 장기 backtest history

## 8. 사고 재발 방지 기준

작업 시작 전에 아래 중 하나라도 아니면 중단한다.

- [ ] 제품 API 조회는 최근 1년 또는 명시 limit 안에 있다.
- [ ] 학습/검증은 Supabase가 아니라 local parquet/cache를 사용한다.
- [ ] upsert 응답은 minimal이다.
- [ ] Render cron suspend 확인 전 전체 sync를 실행하지 않는다.
- [ ] 삭제 전 backup parquet와 checksum이 있다.

## 9. 검증 명령

```powershell
python -m py_compile backend\db\bootstrap.py
python -m unittest backend.tests.test_db_bootstrap
```

이번 CP에서는 위 검증만 허용한다. DB read/write 검증은 별도 승인 후 제한 범위로만 수행한다.
