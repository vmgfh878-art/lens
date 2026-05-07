# Git 정리 계획

생성일: 2026-05-04

## 현재 상태

`git status --short` 기준:

- 총 항목: 148
- tracked 변경: 70
- untracked: 78
- deleted 표시: 2
- 상위 디렉터리 분포: `ai` 76, `backend` 40, `docs` 15, `frontend` 9, `scripts` 7, `logs` 1

`git status --ignored --short` 기준:

- 총 항목: 666
- ignored: 518
- ignored 상위: `docs` 337, `ai` 126, `backend` 20, `logs` 20

이번 계획은 감사 전용이다. 실제 `git add`, `git commit`, `git reset`, 삭제는 하지 않았다.

## 커밋 후보 묶음

### 1. 데이터 provider/source 전환 기반

목적:
- yfinance/EODHD provider abstraction
- source-aware price/indicator 저장
- source mixing guard
- local snapshot 읽기 경로

후보 파일:
- `backend/collector/sources/market_data_providers.py`
- `backend/collector/sources/price_contract.py`
- `backend/collector/jobs/sync_prices.py`
- `backend/collector/jobs/compute_indicators.py`
- `backend/collector/pipelines/yfinance_price_sync.py`
- `backend/collector/pipelines/compute_indicators_cli.py`
- `backend/collector/repositories/local_snapshots.py`
- `backend/db/schema.sql`
- `backend/db/scripts/ensure_runtime_schema.py`
- `backend/tests/test_market_data_providers.py`
- `backend/tests/test_collector_jobs.py`

주의:
- `backend/data/cache/yfinance/*.db*`는 코드가 아니라 로컬 provider cache다. 별도 처리 전까지 이 커밋에 넣지 않는다.

### 2. local parquet/cache/split guard

목적:
- local parquet snapshot 기반 feature/index/split
- source_data_hash와 manifest 분리
- 1W/1M local snapshot 준비
- partial period guard

후보 파일:
- `ai/preprocessing.py`
- `ai/splits.py`
- `ai/ticker_registry.py`
- `ai/tests/test_preprocessing.py`
- `ai/tests/test_preprocessing_cache_isolation.py`
- `ai/tests/test_splits.py`
- `ai/tests/test_ticker_registry.py`
- `scripts/cp98_local_parquet_snapshot_bootstrap.py`
- `scripts/cp100_yfinance_1w_1m_local_snapshot_bootstrap.py`

주의:
- `ai/cache/*.pt`는 이미 ignore 대상이다.
- `ai/cache/*.pt.manifest.json`은 현재 untracked로 보이므로 gitignore 후보에 넣는다.

### 3. Supabase egress/cost guard

목적:
- 대량 read/write 차단
- `returning=minimal` 기본화
- local snapshot required guard
- Supabase slimming 문서화

후보 파일:
- `backend/db/bootstrap.py`
- `backend/collector/repositories/base.py`
- `backend/db/scripts/export_parquet.py`
- `backend/collector/readiness.py`
- `backend/tests/test_db_bootstrap.py`
- `docs/supabase_egress_incident_report.md`
- `docs/supabase_db_slimming_plan.md`
- `docs/supabase_cost_guard_checklist.md`
- `docs/data_architecture_sot.md`
- `docs/data_migration_status_inventory.md`
- `docs/data_contract_risk_register.md`

주의:
- Supabase Usage 실측값은 문서에 원문 secret 없이 숫자만 기록한다.

### 4. 제품 1D loop 및 latest-only 저장

목적:
- CP99/CP101 제품 1D loop
- latest-only prediction thin upload
- 제품 저장 guard

후보 파일:
- `ai/storage.py`
- `ai/tests/test_storage_contracts.py`
- `backend/collector/config.py`
- `backend/tests/test_market_data_providers.py`
- `scripts/cp99_1d_product_loop_thin_upload.py`
- `scripts/cp101_eodhd_off_1d_operation_rehearsal.py`
- `docs/cp99_1d_product_loop_thin_upload_report.md`
- `docs/cp101_eodhd_off_1d_operation_rehearsal_report.md`
- `docs/cp102_product_storage_git_local_cleanup_report.md`

주의:
- `ai.inference --save`는 일반 실험 저장 경로로 남아 있다.
- 제품 운영 명령은 `save_product_latest_predictions()`를 쓰는 thin upload 스크립트로 제한한다.

### 5. 모델/평가/로깅 변경

목적:
- line/band metric 정리
- local training progress logger
- sweep/cache 관련 변경
- 모델 smoke/평가 스크립트

후보 파일:
- `ai/train.py`
- `ai/local_logging.py`
- `ai/evaluation.py`
- `ai/inference.py`
- `ai/backtest.py`
- `ai/sweep.py`
- `ai/composite_inference.py`
- `ai/composite_policy_eval.py`
- `ai/tests/test_local_logging.py`
- `ai/tests/test_inference_backtest.py`
- `ai/tests/test_metric_definition_contract.py`
- `ai/tests/test_sweep.py`
- `ai/tests/test_sweep_caching.py`

주의:
- composite 관련 파일은 제품 기본 경로와 혼동되지 않게 commit message에 legacy/analysis 성격을 명확히 쓴다.

### 6. 프론트/제품 UI 변경

목적:
- 제품 화면/차트/훈련 화면 변경
- line/band layer 표시 변경

후보 파일:
- `frontend/src/api/client.ts`
- `frontend/src/app/globals.css`
- `frontend/src/components/AppShell.tsx`
- `frontend/src/components/StockView.tsx`
- `frontend/src/components/Chart.tsx`
- `frontend/src/components/IndicatorPanel.tsx`
- `frontend/src/components/TrainingView.tsx`
- `frontend/src/components/BacktestView.tsx`

주의:
- CP102에서는 프론트 수정이 금지였으므로, 이 변경들은 기존 dirty worktree 항목으로만 감사했다.
- `frontend/tsconfig.tsbuildinfo`는 빌드 산출물로 보이며 커밋 후보가 아니다.

### 7. 문서/보고서 묶음

목적:
- CP 보고서와 제품/데이터 아키텍처 문서 정리

후보 파일:
- `docs/model_architecture.md`
- `docs/project_journal.md`
- `docs/training_hyperparameters.md`
- `docs/free_market_data_source_migration_audit.md`
- `docs/product_*_review_report.md`
- CP별 보고서와 metrics JSON

주의:
- `.gitignore`에 `docs/cp*`가 있어 CP 보고서 다수가 ignored 상태다. CP 보고서를 추적하려면 ignore 정책을 바꾸거나 `git add -f docs/cp...`가 필요하다. 실제 stage는 하지 않는다.

## 커밋 금지/보류 후보

아래는 코드가 아니라 로컬 산출물 또는 캐시로 본다.

- `ai/cache/*.pt`
- `ai/cache/*.pt.manifest.json`
- `logs/runs/`
- `logs/cp*/`
- `data/parquet/*.parquet`
- `frontend/.next/`
- `frontend/node_modules/`
- `.venv/`
- `wandb/`
- `frontend/tsconfig.tsbuildinfo`
- `backend/data/cache/yfinance/*.db*`

## gitignore 보강 후보

현재 `.gitignore`에는 `*.pt`, `*.parquet`, `backend/data/parquet/`, `logs/cp*/`, `wandb/` 등이 있다. 추가 후보:

```gitignore
ai/cache/*.manifest.json
ai/cache/*.pt.manifest.json
logs/runs/
frontend/tsconfig.tsbuildinfo
backend/data/cache/yfinance/*.db
backend/data/cache/yfinance/*.db-shm
backend/data/cache/yfinance/*.db-wal
```

주의:
- `backend/data/cache/yfinance/*.db*`가 이미 tracked라면 gitignore만으로는 추적이 사라지지 않는다. 별도 승인 후 `git rm --cached` 여부를 결정해야 한다.

## 권장 순서

1. `backend/data/cache/yfinance` tracked cache 처리 방침 결정
2. gitignore 보강
3. 데이터 provider/source 전환 커밋
4. local parquet/cache/split guard 커밋
5. Supabase egress/cost guard 커밋
6. 제품 1D latest loop 커밋
7. 모델/평가/로깅 커밋
8. 프론트 UI 커밋
9. 문서/보고서 커밋

## 위험

- 현재 변경량은 `git diff --stat` 기준 70개 tracked 파일, 약 10100 insertions / 1312 deletions다.
- 한 커밋으로 묶으면 CP별 원인 추적과 rollback이 어렵다.
- `frontend` 변경과 데이터 provider 변경이 섞이면 제품 버그 원인 분리가 어려워진다.
- 캐시 DB와 manifest가 섞이면 재현성 산출물과 로컬 임시 파일의 경계가 흐려진다.
