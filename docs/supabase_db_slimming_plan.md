# CP-SUPA2 Supabase DB 슬리밍 계획

작성일: 2026-05-04 KST  
상태: 계획 확정, 삭제/대량 write 미실행  
목표: Supabase Free 2GB 안에서 운영 가능한 얇은 제품 DB를 만들고, 학습/검증/장기 history는 로컬 parquet/cache로 이전한다.

## 1. Executive Summary

- Supabase에는 제품 화면과 최신 추론에 필요한 최소 데이터만 남긴다.
- 전체 `price_data`, 전체 `indicators`, 학습 feature, 실험용 prediction/evaluation history는 로컬 parquet를 primary로 둔다.
- 이번 CP에서는 DB 삭제, 전체 yfinance write, indicators full recompute, full training, live inference 저장을 실행하지 않았다.
- 용량 원인 분석은 아래 읽기 전용 SQL을 사람이 Supabase SQL Editor에서 실행해 확인한다. `price_data`/`indicators`에 대한 `count(*)` 전체 스캔은 기본 진단에서 금지한다.

## 2. 읽기 전용 용량 추정 SQL

### 2.1 테이블별 전체 용량

```sql
select
  n.nspname as schema_name,
  c.relname as table_name,
  pg_size_pretty(pg_relation_size(c.oid)) as table_size,
  pg_size_pretty(pg_indexes_size(c.oid)) as index_size,
  pg_size_pretty(pg_total_relation_size(c.oid)) as total_size,
  pg_relation_size(c.oid) as table_bytes,
  pg_indexes_size(c.oid) as index_bytes,
  pg_total_relation_size(c.oid) as total_bytes,
  coalesce(s.n_live_tup, c.reltuples)::bigint as estimated_rows,
  s.n_dead_tup as estimated_dead_rows
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
left join pg_stat_user_tables s on s.relid = c.oid
where n.nspname = 'public'
  and c.relkind in ('r', 'p')
  and c.relname in (
    'price_data',
    'indicators',
    'predictions',
    'prediction_evaluations',
    'model_runs',
    'backtest_results',
    'stock_info',
    'company_fundamentals',
    'macroeconomic_indicators',
    'market_breadth',
    'sector_returns',
    'job_runs'
  )
order by pg_total_relation_size(c.oid) desc;
```

### 2.2 기타 대형 테이블 탐색

```sql
select
  n.nspname as schema_name,
  c.relname as table_name,
  pg_size_pretty(pg_total_relation_size(c.oid)) as total_size,
  pg_total_relation_size(c.oid) as total_bytes,
  coalesce(s.n_live_tup, c.reltuples)::bigint as estimated_rows,
  s.n_dead_tup as estimated_dead_rows
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
left join pg_stat_user_tables s on s.relid = c.oid
where n.nspname = 'public'
  and c.relkind in ('r', 'p')
order by pg_total_relation_size(c.oid) desc
limit 30;
```

### 2.3 대형 인덱스 확인

```sql
select
  schemaname,
  tablename,
  indexname,
  pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
  pg_relation_size(indexrelid) as index_bytes
from pg_stat_user_indexes
where schemaname = 'public'
order by pg_relation_size(indexrelid) desc
limit 50;
```

### 2.4 prediction 계열 pruning 후보 추정

이 쿼리는 metadata 중심이며 삭제하지 않는다.

```sql
select
  coalesce(m.status, 'unknown') as run_status,
  coalesce(m.model_name, p.model_name, 'unknown') as model_name,
  coalesce(m.timeframe, p.timeframe, 'unknown') as timeframe,
  count(*) as estimated_prediction_rows,
  min(p.asof_date) as min_asof_date,
  max(p.asof_date) as max_asof_date
from public.predictions p
left join public.model_runs m on m.run_id = p.run_id
group by 1, 2, 3
order by estimated_prediction_rows desc;
```

```sql
select
  coalesce(m.status, 'unknown') as run_status,
  coalesce(m.model_name, e.model_name, 'unknown') as model_name,
  coalesce(m.timeframe, e.timeframe, 'unknown') as timeframe,
  count(*) as estimated_evaluation_rows,
  min(e.asof_date) as min_asof_date,
  max(e.asof_date) as max_asof_date
from public.prediction_evaluations e
left join public.model_runs m on m.run_id = e.run_id
group by 1, 2, 3
order by estimated_evaluation_rows desc;
```

### 2.5 제품 run 보호 목록 확인

제품 run ID는 현재 코드 기준 최소 보호 후보이며, SQL 실행 전 최신 제품 run 계약을 다시 확인한다.

```sql
select
  run_id,
  model_name,
  timeframe,
  horizon,
  status,
  feature_version,
  created_at
from public.model_runs
where run_id in (
  'patchtst-1D-efad3c29d803',
  'cnn_lstm-1D-d0c780dee5e8'
);
```

## 3. 비만 원인 가설

| 우선순위 | 테이블 | 비만 원인 가설 | 판단 방식 |
|---:|---|---|---|
| 1 | `indicators` | ticker x timeframe x 날짜 x source 병렬 row, 36개 이상 numeric 컬럼 | `pg_total_relation_size`, source/timeframe 분포 |
| 2 | `price_data` | 2015년 이후 전체 universe 1D history와 source-aware 병렬 row | table/index bytes, source별 추정 |
| 3 | `predictions` | smoke/full/legacy/composite run별 ticker/date/horizon series 저장 | run_status/model/timeframe group |
| 4 | `prediction_evaluations` | prediction row와 같이 실험별 평가 row 누적 | run_status/model/timeframe group |
| 5 | `company_fundamentals` | Phase 1 제품에는 낮은 우선순위이나 과거 연간/분기 데이터 누적 가능 | total bytes, ticker/date 분포 |
| 6 | `backtest_results` | 실험성 backtest 결과 누적 | run_id/strategy 분포 |

## 4. Supabase 유지 데이터

Supabase는 제품 serving DB로 축소한다.

| 영역 | 유지 범위 | 이유 |
|---|---|---|
| `stock_info` | 전체 ticker 기본 정보 | 검색/표시 필수, 크기 작음 |
| `price_data` | 제품 표시용 최신 구간만 | 프론트 기본 1년 조회와 최신 차트 표시 |
| `indicators` | 제품 표시용 최신 1D 중심, 필요 시 1W/1M 최소 구간 | 차트 보조지표 표시 |
| `model_runs` | 제품 run, 최근 후보 run, 감사상 필요한 요약 | run provenance |
| `predictions` | 제품 run의 최신 rolling 표시 범위 | AI line/band 표시 |
| `prediction_evaluations` | 제품 run의 최신 평가 요약 | 제품 신뢰도 표시 |
| `backtest_results` | 제품 후보 최종 backtest 요약 | 데모/문서 표시 |

## 5. 로컬 parquet 우선 데이터

| 영역 | 로컬 우선 범위 | 보관 위치 |
|---|---|---|
| 전체 `price_data` | 2015년 이후 전체 source history | `data/parquet/price_data_{provider}.parquet` |
| 전체 `indicators` | 전체 timeframe/source indicator history | `data/parquet/indicators_{provider}_{timeframe}.parquet` |
| 학습 feature/cache | feature tensor, feature index, ticker registry | `ai/cache`, `data/parquet` |
| 실험 prediction/evaluation history | 제품 탈락/legacy/composite/failed run | `data/parquet/archive/predictions_*` |
| 실험 backtest history | 비교용 장기 backtest | `data/parquet/archive/backtests_*` |
| 장기 fundamentals | Phase 1 제품 미사용 장기 재무 | `data/parquet/company_fundamentals.parquet` |

## 6. pruning 정책 제안

### 6.1 절대 유지

- 제품 line run: `patchtst-1D-efad3c29d803`
- 제품 band run: `cnn_lstm-1D-d0c780dee5e8`
- 제품 화면이 참조하는 최신 prediction/evaluation/backtest row
- `stock_info`
- 현재 제품 표시를 위한 최신 가격/보조지표 구간

### 6.2 백업 후 삭제 후보

- `model_runs.status in ('failed_nan', 'failed_quality_gate')`
- composite legacy run 및 `deprecated_for_phase1_product_contract=true`인 run
- CP smoke run, no-save 정책 이전에 남은 실험성 run
- 제품 run이 아닌 오래된 `predictions`
- 제품 run이 아닌 오래된 `prediction_evaluations`
- 발표/제품에서 쓰지 않는 `backtest_results`

### 6.3 삭제 전 백업 절차

1. 삭제 후보 SQL을 먼저 `select`로 검증한다.
2. 후보 row를 제한 조건별 parquet로 export한다.
3. export 파일마다 `row_count`, `min/max date`, `run_id count`, SHA256 checksum을 기록한다.
4. parquet를 다시 read해서 row_count/checksum 일치 확인한다.
5. 제품 run 보호 목록과 anti-join 검사를 한다.
6. 사용자 승인 후 별도 CP에서 delete를 실행한다.
7. delete 후 `VACUUM`은 Supabase 관리 제약을 고려해 대시보드/SQL 가능 범위에서 별도 확인한다.

삭제 후보 확인 예시:

```sql
select p.*
from public.predictions p
left join public.model_runs m on m.run_id = p.run_id
where p.run_id not in ('patchtst-1D-efad3c29d803', 'cnn_lstm-1D-d0c780dee5e8')
  and (
    coalesce(m.status, '') in ('failed_nan', 'failed_quality_gate')
    or coalesce(m.config->>'deprecated_for_phase1_product_contract', 'false') = 'true'
    or coalesce(m.config->>'role', '') in ('composite_legacy', 'smoke', 'experiment')
  );
```

## 7. upsert egress 방지

PostgREST는 upsert 응답으로 row representation을 돌려줄 수 있다. 대량 upsert에서는 이 응답이 egress가 되므로 `return=minimal`이 기본이어야 한다.

이번 CP에서 `backend/db/bootstrap.py`의 `chunked_upsert()` 기본값을 `ReturnMethod.minimal`로 고정했다. 호출자가 소량 dry-run에서 representation이 필요하면 명시 override만 허용한다.

대량 upsert 전 gate:

- `returning=ReturnMethod.minimal` 기본값 확인
- `--write`가 있는 명령은 ticker/date 범위 제한 확인
- `price_data`/`indicators` 전체 universe write 금지 여부 확인
- Render cron suspend 여부 확인
- 로컬 parquet backup/manifest 확인

## 8. Render cron suspend 확인

사람이 직접 확인해야 한다.

1. Render Dashboard 접속
2. Cron Jobs에서 `lens-daily-market-sync` 선택
3. 상태가 Suspended/Paused인지 확인
4. 확인 전에는 아래 실행 금지:
   - `python -m backend.collector.pipelines.daily_market_sync`
   - `python -m backend.collector.pipelines.daily_sync`
   - `scripts/run_daily_local_market_sync.ps1` write mode
   - `scripts/sync_yfinance_prices.ps1 -Write`

`render.yaml` 기준 현재 cron:

```yaml
name: lens-daily-market-sync
startCommand: python -m backend.collector.pipelines.daily_market_sync --indicator-lookback-days 60
```

## 9. 다음 CP 제안

1. Supabase SQL Editor에서 용량 추정 SQL만 실행하고 결과를 `docs/supabase_db_size_snapshot_YYYYMMDD.md`로 저장한다.
2. 제품 run 보호 목록을 확정한다.
3. pruning 후보 parquet export 스크립트를 제한 조건 기반으로 만든다.
4. 사용자 승인 후 predictions/evaluations/backtests부터 삭제 CP를 분리한다.
5. 마지막에 price_data/indicators 최신 구간 retention 정책을 적용한다.
