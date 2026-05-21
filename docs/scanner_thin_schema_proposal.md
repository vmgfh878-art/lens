# Scanner Thin Schema 제안

작성일: 2026-05-06  
상태: 제안 전용. 이번 CP에서 DB schema 변경은 하지 않았다.

## 1. 목적

500티커 전체 inference 결과를 Supabase에 모두 저장하지 않고, 로컬 parquet에 full raw prediction을 보관한 뒤 제품/API에는 top-k scanner 신호만 얇게 노출한다.

핵심 원칙:
- full 500 raw prediction: local parquet/archive only
- Supabase: scanner run 요약 + top-k signal + 필요 시 selected ticker latest prediction
- API: top-k만 기본 반환
- price_data/indicators 전체 history 업로드 금지

## 2. scanner_runs

제안 테이블:

```sql
create table public.scanner_runs (
  scanner_run_id uuid primary key default gen_random_uuid(),
  scanner_name text not null,
  timeframe varchar(4) not null check (timeframe in ('1D', '1W', '1M')),
  asof_date date not null,
  source text not null,
  provider text not null,
  provider_adjustment_policy text,
  feature_version text not null,
  source_data_hash text not null,
  line_run_id text,
  band_run_id text,
  universe_count integer not null,
  eligible_count integer not null,
  evaluated_count integer not null,
  selected_count integer not null,
  top_k integer not null,
  score_policy text not null,
  status text not null check (status in ('success', 'failed', 'partial')),
  started_at timestamptz,
  finished_at timestamptz,
  elapsed_seconds double precision,
  local_artifact_uri text,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (scanner_name, timeframe, asof_date, source, score_policy)
);

create index idx_scanner_runs_latest
  on public.scanner_runs (timeframe, asof_date desc, created_at desc);
```

필수 meta:
- `local_prediction_path`
- `full_prediction_row_count`
- `full_prediction_checksum`
- `ranked_universe_checksum`
- `fallback_used_count`
- `bulk_upload_blocked=true`

## 3. scanner_signals

제안 테이블:

```sql
create table public.scanner_signals (
  id bigserial primary key,
  scanner_run_id uuid not null references public.scanner_runs(scanner_run_id) on delete cascade,
  rank integer not null,
  ticker varchar(20) not null references public.stock_info(ticker),
  timeframe varchar(4) not null check (timeframe in ('1D', '1W', '1M')),
  asof_date date not null,
  signal text not null check (signal in ('buy', 'watch', 'risk', 'reentry', 'hold')),
  score double precision not null,
  score_components jsonb not null default '{}'::jsonb,
  line_run_id text,
  band_run_id text,
  horizon integer not null,
  forecast_start_date date,
  forecast_end_date date,
  conservative_return double precision,
  lower_return double precision,
  upper_return double precision,
  band_width_return double precision,
  downside_risk_score double precision,
  upside_score double precision,
  trend_score double precision,
  volatility_score double precision,
  rsi double precision,
  atr_ratio double precision,
  ma60_ratio double precision,
  local_prediction_ref text,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (scanner_run_id, ticker)
);

create index idx_scanner_signals_rank
  on public.scanner_signals (scanner_run_id, rank);

create index idx_scanner_signals_latest_ticker
  on public.scanner_signals (ticker, timeframe, asof_date desc);
```

저장 정책:
- `rank <= top_k`만 저장한다.
- 기본 `top_k=50`, 최대 `top_k=100`.
- forecast full series는 저장하지 않는다. 카드/목록에 필요한 summary 값만 저장한다.
- 전체 500 prediction series는 `local_prediction_ref`로만 참조한다.

## 4. selected ticker latest prediction 저장

기본 정책:
- scanner top-k 중 차트 상세 표시가 필요한 ticker만 기존 `predictions`에 latest-only로 저장한다.
- line/band는 각각 별도 row로 저장한다.
- composite 저장 금지.
- meta에 아래 값을 넣는다.

```json
{
  "storage_contract": "scanner_selected_latest_only",
  "scanner_run_id": "...",
  "scanner_rank": 1,
  "layer": "line",
  "source": "yfinance",
  "provider": "yfinance",
  "thin_upload": true
}
```

주의:
- 현재 `predictions` unique key는 `run_id,ticker,model_name,timeframe,horizon,asof_date`라서 asof_date가 바뀌면 row가 누적된다.
- 진짜 latest-only를 보장하려면 별도 pruning CP 또는 `product_latest_predictions`류 테이블이 필요하다.
- CP126에서는 schema 변경을 하지 않았으므로, selected latest 저장 확대 전 pruning/retention gate가 먼저 필요하다.

## 5. API 제안

기본 API:
- `GET /api/v1/scanner/runs/latest?timeframe=1D`
- `GET /api/v1/scanner/runs/{scanner_run_id}/signals?limit=50`
- `GET /api/v1/scanner/signals/latest?timeframe=1D&limit=50`

응답 기본값:
- top-k만 반환
- 기본 limit 50, 최대 100
- `score_components`는 포함하되 full prediction series는 제외
- 상세 차트는 기존 `/stocks/{ticker}/predictions/latest`를 사용한다.

## 6. 보존 정책

Supabase:
- `scanner_runs`: 1D 최근 90거래일, 1W 최근 52주
- `scanner_signals`: run별 top-k만 보존
- `predictions`: selected ticker latest-only 기본
- `prediction_evaluations`: forward-only scanner 단계에서는 저장하지 않음. 실제값 확정 후 별도 evaluator에서 요약만 저장

Local:
- full 500 prediction parquet: 최근 1년 보존
- scanner ranked universe parquet: 최근 1년 보존
- 오래된 raw prediction은 월 단위 archive 후 삭제 후보

## 7. Cutover Gate

scanner schema를 실제 반영하기 전 필요한 조건:
- 500티커 local price/indicator snapshot 생성
- yfinance append fallback 0 또는 실패 ticker exception list 생성
- 1D/1W partial period guard 통과
- 500티커 inference dry-run payload 생성
- top-k row 수와 payload 크기 측정
- Supabase `returning=minimal` upsert 확인
- pruning dry-run과 rollback/export 절차 문서화
