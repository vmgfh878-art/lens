# CP117-DG 로컬 Parquet + Supabase 얇은 DB 운영 계약 마감

생성일: 2026-05-06

## 1. 요약

최종 판정: PASS

로컬 Parquet를 원천 데이터 저장소로, Supabase를 제품 표시용 얇은 DB로 쓰는 구조를 운영 기준으로 고정했다. Supabase는 더 이상 전체 `price_data`/`indicators`/실험 이력 창고가 아니며, 제품 실행의 최신 prediction/evaluation과 최소 메타만 유지한다.

이번 CP에서는 DB 삭제/쓰기를 하지 않았다. Supabase REST `count=exact + limit=1`로 행 수 예행 점검만 수행했고, `price_data`/`indicators` 원문 대량 읽기는 가드로 차단됨을 확인했다.

## 2. 현재 Supabase 행 수

| table | row count | status |
|---|---:|---|
| `price_data` | 1,674,722 | PASS |
| `indicators` | 1,929,811 | PASS |
| `predictions` | 360,773 | PASS |
| `prediction_evaluations` | 360,773 | PASS |
| `model_runs` | 22 | PASS |
| `job_runs` | 121 | PASS |
| `stock_info` | 507 | PASS |

주의:
- 위 숫자는 row count다.
- 실제 byte 용량은 Supabase SQL Editor에서 `pg_total_relation_size` SQL을 실행해야 한다.
- CP117 스크립트는 임의 SQL 실행 권한이 없어 table size SQL을 자동 실행하지 않았다.

## 3. 실제 용량 측정 SQL

아래 SQL은 Supabase SQL Editor에서 실행한다.

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
    'job_runs',
    'stock_info'
  )
order by pg_total_relation_size(c.oid) desc;
```

실행 가능 여부:
- SQL 자체는 준비 완료
- script 자동 실행: 미실행
- 이유: Supabase REST client에는 임의 SQL 실행 기능이 없고, 별도 RPC 함수가 없다.

## 4. 제품 최신값 전용 저장 범위

제품 run:

| layer | run_id |
|---|---|
| line | `patchtst-1D-efad3c29d803` |
| band | `cnn_lstm-1D-d0c780dee5e8` |

현재 제품 행 수:

| table | product total | product latest | product history excess | non-product |
|---|---:|---:|---:|---:|
| `predictions` | 357,484 | 10 | 357,474 | 3,289 |
| `prediction_evaluations` | 357,484 | 10 | 357,474 | 3,289 |

정책:
- Supabase 기본 저장 범위는 제품 최신값 전용이다.
- rolling history는 local parquet/archive에 둔다.
- Supabase rolling history가 필요하면 최근 N개 asof를 명시적으로 제한한다.
- composite 저장은 금지한다.

## 5. 가지치기 예행 점검

삭제 실행: 없음

삭제 후보 row count:

| group | candidate rows |
|---|---:|
| `predictions_non_product` | 3,289 |
| `predictions_product_history_except_latest` | 357,474 |
| `prediction_evaluations_non_product` | 3,289 |
| `prediction_evaluations_product_history_except_latest` | 357,474 |
| `model_runs_non_product` | 20 |
| `model_runs_failed_quality` | 9 |

합계:
- price/indicator 제외 가지치기 후보: 721,555 rows

주의:
- `model_runs_non_product`와 `model_runs_failed_quality`는 서로 겹칠 수 있으므로 실제 delete SQL에서는 중복 제거가 필요하다.
- product run 2개는 보호한다.
- deletion은 backup/export/checksum 후 별도 승인 CP에서만 수행한다.

## 6. 가지치기 예행 점검 SQL

```sql
with product_runs as (
  select unnest(array[
    'patchtst-1D-efad3c29d803',
    'cnn_lstm-1D-d0c780dee5e8'
  ]) as run_id
),
latest_product_predictions as (
  select p.run_id, max(p.asof_date) as latest_asof_date
  from public.predictions p
  join product_runs pr on pr.run_id = p.run_id
  group by p.run_id
),
latest_product_evaluations as (
  select e.run_id, max(e.asof_date) as latest_asof_date
  from public.prediction_evaluations e
  join product_runs pr on pr.run_id = e.run_id
  group by e.run_id
)
select
  'predictions_non_product' as candidate_group,
  count(*) as candidate_rows
from public.predictions p
where not exists (select 1 from product_runs pr where pr.run_id = p.run_id)
union all
select
  'predictions_product_history_except_latest' as candidate_group,
  count(*) as candidate_rows
from public.predictions p
join latest_product_predictions latest on latest.run_id = p.run_id
where p.asof_date < latest.latest_asof_date
union all
select
  'prediction_evaluations_non_product' as candidate_group,
  count(*) as candidate_rows
from public.prediction_evaluations e
where not exists (select 1 from product_runs pr where pr.run_id = e.run_id)
union all
select
  'prediction_evaluations_product_history_except_latest' as candidate_group,
  count(*) as candidate_rows
from public.prediction_evaluations e
join latest_product_evaluations latest on latest.run_id = e.run_id
where e.asof_date < latest.latest_asof_date;
```

## 7. 로컬 Snapshot 상태

| snapshot | rows | latest date | tickers |
|---|---:|---|---:|
| `stock_info.parquet` | 100 | 없음 | 100 |
| `price_data_yfinance.parquet` | 284,905 | 2026-05-04 | 100 |
| `indicators_yfinance_1D.parquet` | 279,005 | 2026-05-04 | 100 |
| `price_data_yfinance_1W.parquet` | 59,200 | 2026-05-01 | 100 |
| `indicators_yfinance_1W.parquet` | 53,300 | 2026-05-01 | 100 |

## 8. 얇은 DB 유지 정책

정책 문서:
- `docs/supabase_thin_retention_policy.md`

핵심:
- `stock_info`: Supabase 유지
- `model_runs`: 제품 run + 최근 completed 실험 일부만 유지
- `predictions`: 제품 최신값 전용 기본
- `prediction_evaluations`: 제품 최신값 전용 기본
- rolling history: local parquet 기본
- `price_data`/`indicators`: Supabase 전체 저장소 역할 제거

price/indicator 판단:
- 최종 목표는 backend/API가 local parquet를 읽고 Supabase에는 원본 full history를 두지 않는 구조다.
- 배포 제약 때문에 Supabase price/indicator가 필요하면 활성 제품 universe의 최근 1년 정도로 제한한다.

## 9. 일일 운영 경계

상세 문서:
- `docs/local_parquet_supabase_sync_boundary.md`

순서:
1. yfinance 완료 거래일만 로컬 가격 Parquet에 추가
2. 추가된 ticker만 1D indicator 증분 갱신
3. 로컬 가격 최신일 == 로컬 indicator 최신일 확인
4. 1D line/band latest inference
5. 제품 최신값 전용 얇은 업로드
6. API latest asof_date 확인

중단 조건:
- source/hash/asof_date mismatch
- EODHD fallback 발생
- Supabase bulk read guard 실패
- partial current-day row append 시도
- composite 저장 시도

## 10. 준비도와 최신성 체크

필수 체크:
- local price latest date
- local indicator latest date
- product prediction latest asof_date
- Supabase latest asof_date
- line/band run_id
- EODHD fallback 사용 여부 0
- Supabase bulk read guard

CP117 기준:
- expected current 1D asof: `2026-05-04`
- bulk read guard: `price_data=PASS_BLOCKED`, `indicators=PASS_BLOCKED`

## 11. 금지 항목 확인

| 항목 | 발생 |
|---|---:|
| 전체 yfinance write | False |
| indicators full recompute | False |
| 모델 학습 | False |
| inference 대량 저장 | False |
| row delete | False |
| EODHD call | False |
| DB write | False |

## 12. 최종 판정

PASS.

Supabase thin 유지 정책이 문서로 고정됐고, product latest-only 저장 범위가 명확해졌다. pruning은 아직 실행하지 않았지만, 삭제 후보 row count와 백업/복구 절차가 정리됐다. 1W 모델 실험 전 운영 boundary는 설명 가능하다.
