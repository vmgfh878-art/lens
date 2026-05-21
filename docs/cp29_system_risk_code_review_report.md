# CP29 Lens 전체 시스템 리스크 코드 리뷰 보고서

작성일: 2026-04-29

범위: 데이터 수집/동기화, 피처/캐시, 학습, 평가/백테스트, 백엔드 API, 프론트/데모 안정성, 운영/보안/저장소

제약: 코드 수정 금지, 테스트 수정 금지, 포맷팅 금지. 본 파일만 신규 작성.

## 1. Executive Summary

현재 코드 기준으로 과거의 대형 병목 일부는 이미 개선되어 있다. `prepare_dataset_splits()`는 lazy `SequenceDataset`을 사용하고, `ai/sweep.py`는 trial마다 데이터셋을 다시 만들지 않고 shared bundle을 넘긴다. `train.py`에도 gradient clipping, scheduler, early stopping, batch 기반 평가, band postprocess가 들어와 있다.

다만 모델 실험을 재개하기 전에 막아야 할 P1은 아직 남아 있다. 가장 큰 축은 DB 저장 계약 불일치, prediction run 재현성, ticker embedding registry 재현성, PatchTST RevIN 출력 스케일, backtest 수익률 계산 계약이다. 이 상태로 `--save-run`, inference `--save`, backtest를 신뢰 지표로 쓰면 모델 성능이 아니라 저장/조회/스케일 오류를 평가할 가능성이 크다.

데이터 재건 쪽은 adjusted OHLC 계약 자체는 코드에 들어와 있지만, cache fingerprint와 model run `feature_version`이 실제 feature contract를 충분히 대표하지 못한다. macro/breadth의 release calendar 부재, fundamentals restatement/as-of snapshot 부재는 아직 point-in-time 리스크로 남아 있다.

## 2. P0/P1/P2/P3 리스크 표

| 우선순위 | ID | 리스크 | 근거 | 영향 | 비고 |
|---|---|---|---|---|---|
| P0 | P0-01 | 확정 P0 없음 | 전체 감사 범위에서 즉시 데이터 삭제/보안키 노출/실행 불능 확정 건은 발견하지 못함 | 없음 | 현재 P1들이 모델 결과 신뢰성을 막는 수준 |
| P1 | P1-01 | `model_runs` 저장 계약이 현재 코드와 스키마가 다름 | `ai/train.py:1478-1488`은 `band_mode`를 top-level로 저장하지만 `backend/db/schema.sql:210-233`, `backend/db/scripts/ensure_runtime_schema.py:84-147`에는 해당 컬럼 추가가 없음 | fresh DB 또는 runtime schema 기준이면 학습 종료 후 `save_run`이 실패할 수 있음 | 실제 DB에 수동 컬럼이 있으면 당장 통과할 수 있으나 repo 기준 불일치 |
| P1 | P1-02 | `prediction_evaluations` 저장 레코드에 스키마 없는 컬럼이 포함됨 | `ai/evaluation.py:174-177`, `ai/inference.py:282-297`은 `lower_breach_rate`, `upper_breach_rate`를 만들고 `ai/storage.py:32-36`은 그대로 upsert. 하지만 `backend/db/schema.sql:239-254`, `ensure_runtime_schema.py:110-125`에는 컬럼 없음 | inference `--save`가 DB에서 실패하거나 실제 DB와 repo schema가 갈라짐 | 저장 전 필수 수정 대상 |
| P1 | P1-03 | prediction upsert key가 `run_id`를 제외해 과거 run 결과를 덮음 | `backend/db/schema.sql:183-202`, `ai/storage.py:17-23` | 같은 ticker/model/timeframe/horizon/asof_date의 새 run이 이전 run의 prediction row와 `run_id`를 덮어, `fetch_run_predictions(run_id)`와 backtest 재현성이 깨짐 | run_id별 결과 테이블 계약 필요 |
| P1 | P1-04 | inference에서 checkpoint의 ticker registry를 읽지 않고 현재 universe로 ticker id를 재생성 | `ai/train.py:1136-1137`은 registry path를 저장하지만, `ai/inference.py:109-136`, `ai/preprocessing.py:1215-1222`는 현재 eligible ticker로 `build_registry()`를 다시 수행 | 종목 universe, 필터, DB 상태가 바뀌면 embedding ID 의미가 바뀜 | ticker embedding 모델의 예측 의미가 붕괴 가능 |
| P1 | P1-05 | PatchTST RevIN denormalize가 타깃 스케일과 다름 | `ai/models/patchtst.py:93-95`, `ai/models/revin.py:48-59`, `ai/preprocessing.py:154-181` | 출력은 미래 누적수익률인데 RevIN은 입력 `target_channel_idx=0`의 과거 feature 통계로 복원한다. 입력 0번은 표준화된 과거 `log_return` 계열이라 raw future return과 같은 시계열이 아님 | 설계 검토 전 PatchTST 성능 해석 위험 |
| P1 | P1-06 | backtest가 adjusted target/prediction을 raw close anchor로 다시 수익률화함 | target은 `ai/preprocessing.py:746-754`에서 `adjusted_close` 기반, inference actual/line price는 `ai/inference.py:224-296`에서 anchor 기반. backtest는 `ai/backtest.py:46-71`에서 `price_data.close`를 anchor로 사용 | split/dividend 이벤트가 있으면 realized/line return이 왜곡됨 | 가격 API도 raw close라 overlay와 같은 계열 정렬 필요 |
| P1 | P1-07 | backtest가 여러 ticker 행을 날짜별 포트폴리오로 묶지 않고 행 단위 시계열로 누적 | `ai/backtest.py:85-105`에서 전체 frame sort 후 `position.shift(1)`, row-level `cumprod()` 사용 | 한 ticker의 포지션이 다른 ticker의 previous position처럼 이어지고, 수수료/샤프/누적수익이 단면 포트폴리오가 아니라 행 순서에 의존 | 투자 지표로 사용 금지 |
| P2 | P2-01 | latest prediction 조회가 `model_runs.status`를 확인하지 않음 | run_id 조회는 `backend/app/services/api_service.py:95-105`에서 status 확인. latest 조회는 `api_service.py:109-119`, `backend/app/repositories/prediction_repo.py:14-37`에서 prediction만 조회 | non-completed run prediction이 남아 있거나 run 상태가 바뀌면 최신 endpoint가 무효 결과를 반환 가능 | `failed_nan`은 저장 차단되지만 일반 방어가 부족 |
| P2 | P2-02 | feature cache fingerprint가 데이터 내용과 feature 로직 변경을 충분히 반영하지 않음 | `ai/preprocessing.py:624-666`은 `price_max_date`, `indicator_max_date`, `indicator_count` 중심. cache key는 `ai/preprocessing.py:678-743`에서 수동 `_FEATURE_CONTRACT_VERSION`에 의존 | 같은 row count/max date에서 값이 재계산되거나 로직이 바뀌면 stale cache 학습 가능 | 데이터 재건 후 cache 무효화 필요 |
| P2 | P2-03 | model run `feature_version`이 실제 feature contract와 다름 | `_FEATURE_CONTRACT_VERSION="v3_adjusted_ohlc"`는 `ai/preprocessing.py:51`, 학습 저장은 `ai/train.py:1224`, `ai/train.py:1478`에서 `"indicators_v1"` | run metadata만 보고 어떤 feature contract로 학습됐는지 알 수 없음 | 실험 비교와 재현성 훼손 |
| P2 | P2-04 | coverage gate 실패 시에도 completed run으로 저장될 수 있음 | `ai/train.py:327-345`, `ai/train.py:1375-1391`, `ai/train.py:1471-1505` | `checkpoint_selection=coverage_gate`가 hard gate가 아니라 val_total fallback이므로, gate 실패 모델이 completed로 남음 | 제품/데모 run 선택 기준에 혼입 가능 |
| P2 | P2-05 | checkpoint 로드가 feature schema/registry content를 검증하지 않음 | `ai/inference.py:68-93`은 현재 `MODEL_N_FEATURES`로 모델을 만들고 state_dict만 로드 | feature columns, calendar columns, registry mapping이 바뀌어도 사전 오류 메시지가 없고 shape mismatch 또는 조용한 의미 불일치 가능 | checkpoint에 schema hash와 registry snapshot 필요 |
| P2 | P2-06 | 평가/추론 summary가 batch forward 후에도 전체 split prediction을 `torch.cat`으로 모음 | `ai/train.py:831-890`, `ai/inference.py:176-208`, `ai/inference.py:301-311` | GPU OOM은 줄었지만 full universe/긴 horizon에서 CPU 메모리와 latency가 커짐 | streaming metric 누적 필요 |
| P2 | P2-07 | 물리화 dataset 경로가 아직 남아 있음 | `ai/preprocessing.py:757-899` | 현재 `prepare_dataset_splits()`는 lazy 경로를 쓰지만, 누군가 `build_sequence_dataset()`을 직접 호출하면 전체 tensor 물리화 리스크가 재발 | legacy path 경고 또는 제거 대상 |
| P2 | P2-08 | EODHD 수집 실패가 빈 DataFrame으로 숨겨짐 | `backend/collector/sources/eodhd_prices.py:24-44`, `backend/collector/sources/eodhd_prices.py:78-91` | 인증, 네트워크, schema 오류가 `source_empty` 또는 Yahoo fallback으로 섞여 원인 추적이 늦어짐 | source/error reason 저장 필요 |
| P2 | P2-09 | daily sync와 indicator 계산이 partial/zero 결과를 success로 남길 수 있음 | `backend/collector/jobs/sync_prices.py:201-260`, `backend/collector/jobs/compute_indicators.py:109-207`, `backend/collector/pipelines/daily_sync.py:91-164` | quota_hit 또는 일부 timeframe 0건이어도 상위 job이 success가 될 수 있음 | market-only는 coverage gate가 있으나 full daily path는 약함 |
| P2 | P2-10 | macro/breadth가 publication/release date 기준 as-of가 아님 | `backend/app/services/feature_svc.py:258-272`, `backend/app/services/feature_svc.py:495-507` | observation date 기준 merge/ffill이면 FRED류 release lag 누수 가능 | 추측 포함. release calendar 테이블 부재 기준 |
| P2 | P2-11 | fundamentals는 filing_date merge는 하지만 restatement/as-of snapshot이 없음 | `backend/app/services/feature_svc.py:404-428`, `backend/collector/jobs/sync_fundamentals.py:87`, `backend/collector/jobs/sync_fundamentals.py:132`, `backend/collector/jobs/sync_fundamentals.py:176` | 같은 fiscal date가 최신 source 값으로 overwrite되면 과거 학습 시점에 몰랐던 restated 값이 들어갈 수 있음 | 추측. source가 historical filing snapshot을 주는지 확인 필요 |
| P2 | P2-12 | stock_info 503개 원인 후보가 placeholder seed일 수 있음 | `backend/collector/jobs/sync_stock_info.py:14-23`, `backend/collector/jobs/sync_stock_info.py:33-48` | universe에 있는 ticker를 먼저 placeholder로 넣기 때문에 metadata 실패와 무관하게 row count가 늘 수 있음 | 여기서 503은 HTTP 503이 아니라 row count라는 추측 |
| P2 | P2-13 | stock search는 ticker만 검색하고 upstream 오류는 503으로 감싸짐 | `backend/app/repositories/market_repo.py:5-31`, `backend/app/routers/v1/stocks.py:23-36`, `backend/app/core/exceptions.py:24-31` | 회사명 검색 불가, Supabase 문제는 모두 “종목 목록 조회 실패” 503으로 보임 | 사용자 영향은 검색 실패/빈 목록 |
| P2 | P2-14 | Backtest 화면은 latest completed run만 보고 usable run scan을 하지 않음 | `frontend/src/components/BacktestView.tsx:77-111`. 반면 StockView는 `frontend/src/components/StockView.tsx:229-304`에서 run 후보를 순회 | 최신 run에 backtest row가 없으면 이전 usable run이 있어도 empty state | demo readiness script는 보완하지만 화면은 취약 |
| P2 | P2-15 | Optuna summary가 completed trial 0개일 때 실패 가능 | `ai/sweep.py:200-214` | 모든 trial이 pruned/failed이면 `study.best_trial` 접근에서 summary 생성 실패 | HPO 진단성이 나빠짐 |
| P2 | P2-16 | ticker registry 파일이 생성물인데 git tracked이고 plan 생성이 덮어씀 | `ai/ticker_registry.py:10-40`, `ai/preprocessing.py:1003-1036`, `git ls-files` 결과 `ai/cache/ticker_id_map_1d.json`, `ticker_id_map_1w.json` 추적 | dry-run/진단 호출이 registry 파일을 바꾸고, 학습 run별 mapping 재현성을 흐림 | registry는 run artifact로 고정 필요 |
| P2 | P2-17 | README 학습 예시가 표준 1D `seq_len=252`를 적용하지 않음 | `README.md:116-120`, `ai/train.py:363` | README대로 돌리면 `seq_len=60`으로 학습되어 계획서의 1D 252와 다름 | 실험 재현성 문서 리스크 |
| P3 | P3-01 | 문서 일부가 최신 코드와 충돌하거나 과장됨 | `docs/model_architecture.md:17-20`은 구버전 결함을 현재형으로 적고, `README.md:147-150`은 FRED publication date as-of를 명시하지만 코드에는 release calendar가 없음 | 새 작업자가 잘못된 상태 인식으로 재개 가능 | 현재 문서 재건 작업 중이므로 보고만 함 |
| P3 | P3-02 | `docs/cp*`가 gitignore라 감사/실험 보고서가 기본적으로 추적되지 않음 | `.gitignore:4`, `git check-ignore -v docs\cp29_system_risk_code_review_report.md` | 중요한 발표/감사 기록이 untracked 상태로 남아 공유 누락 가능 | 본 보고서도 ignored 대상 |
| P3 | P3-03 | demo script는 `.next` 손상 복구를 직접 수행하지 않음 | `scripts/start_demo.ps1:21-30`, `scripts/check_demo_readiness.ps1:140-275`, `.gitignore:14` | `.next` 캐시가 깨진 상태면 start_demo가 그대로 실패할 수 있음 | check script는 상태 확인만 함 |
| P3 | P3-04 | Windows/PowerShell 실행 스크립트 의존성이 큼 | `scripts/start_demo.ps1`, `scripts/check_demo_readiness.ps1`, collector 실행 예시 | Windows 로컬 데모에는 맞지만 CI/Linux 재현성은 낮음 | 현재 사용자 환경이 Windows라 즉시 차단 아님 |
| P3 | P3-05 | 대용량 cache/checkpoint가 로컬에 누적됨 | `ai/cache` 41개 약 3.5GB, `ai/artifacts` 105개 약 236MB, `.gitignore:16-17` | 디스크와 stale cache 혼선 리스크 | 삭제 정책은 별도 결정 필요 |

## 3. 지금 즉시 막아야 할 것

1. `--save-run` 본학습 저장을 잠시 막아야 한다. `model_runs.band_mode` 스키마 불일치가 repo 기준으로 남아 있어 학습 완료 후 저장 단계에서 실패할 수 있다.

2. inference `--save`를 막아야 한다. `prediction_evaluations`에 없는 `lower_breach_rate`, `upper_breach_rate`를 저장하려는 경로가 있어 DB 저장 계약이 먼저 맞아야 한다.

3. run별 prediction 재현성 없이 backtest를 새로 돌리면 안 된다. `predictions` unique key에 `run_id`가 빠져 있어 새 run 저장이 과거 run의 prediction을 덮을 수 있다.

4. ticker embedding을 쓰는 checkpoint는 현재 inference 결과를 신뢰하면 안 된다. checkpoint registry mapping을 로드하지 않고 현재 DB 기준으로 mapping을 재생성한다.

5. PatchTST RevIN denormalize는 설계 결론 전까지 full run 판단 기준에서 제외하는 편이 안전하다. 현재 복원 통계와 타깃 의미가 다르다.

6. `ai/backtest.py` 결과는 투자 지표로 쓰지 말아야 한다. adjusted/raw anchor mismatch와 행 단위 portfolio 누적 문제가 동시에 있다.

## 4. 데이터 재건 이후 확인할 것

1. adjusted/raw 가격 계약: `price_data` raw OHLC와 `adjusted_close`, feature adjusted OHLC, target adjusted close, API chart close가 같은 목적에 맞게 분리되는지 확인한다.

2. stock_info 503 원인: universe 파일 ticker 수, placeholder seed row, 실제 metadata 성공 row를 분리 집계한다. sector/industry/market_cap null 비율도 같이 봐야 한다.

3. cache 무효화: 데이터 재건 후 `ai/cache/features_*`, `feature_index_*`가 새 contract로만 재생성되는지 확인한다. 현재 fingerprint는 content hash가 아니므로 수동 삭제 또는 contract bump가 필요하다.

4. macro/breadth PIT: FRED publication/release date, market breadth 계산 기준일, forward fill 가능 범위를 분리한다. release date가 없으면 문서의 publication date as-of 표현은 내려야 한다.

5. fundamentals PIT: `company_fundamentals`에 source, fiscal period date, filing_date, 수집일/as_of, restatement 여부를 분리할 수 있는지 확인한다.

6. 1D/1W/1M 정합성: feature_svc는 1M 생성이 가능하지만 AI는 1D/1W만 지원한다. 1M은 가격 전용이라는 API/프론트/문서 계약을 계속 유지해야 한다.

## 5. 모델 실험 재개 전 필수 게이트

1. DB schema smoke: `save_model_run`, `save_predictions`, `save_prediction_evaluations`, `save_backtest_results`를 1건씩 실제 DB에 넣고 지우지 않는 방식의 dry-run 또는 test namespace로 검증한다.

2. ticker registry lock: checkpoint에 registry content hash와 mapping snapshot을 넣고, inference는 해당 mapping을 강제로 로드해야 한다.

3. feature schema lock: checkpoint/config/model_runs에 `SOURCE_FEATURE_COLUMNS`, `CALENDAR_FEATURE_COLUMNS`, `_FEATURE_CONTRACT_VERSION`, data fingerprint를 저장하고 inference에서 비교해야 한다.

4. RevIN target gate: raw future return 예측에서는 output denormalize를 끄거나, 동일 시계열 target 복원 설계로 바꾼 뒤 smoke를 통과해야 한다.

5. backtest gate: adjusted anchor 기준, 날짜별 단면 포트폴리오 수익률, 수수료 turnover, long/short weights를 하나의 계약으로 다시 고정해야 한다.

6. coverage gate hard policy: coverage gate 실패 run을 `completed`로 남길지, `failed_quality_gate` 같은 별도 status로 남길지 결정해야 한다.

7. memory/time gate: full 1D 473 ticker 기준 dataset build, 1 epoch, validation, inference save가 peak RAM/VRAM/시간 로그를 남기고 통과해야 한다.

8. doc gate: README의 `seq_len`, feature list, FRED publication date, 모델 구조 설명을 현재 코드와 맞춘 뒤 실험 지시서에 사용해야 한다.

## 6. 수정 금지 파일을 건드리지 않았다는 확인

수정 금지 파일은 읽기 전용으로만 확인했다.

수정 금지 파일:

- `ai/preprocessing.py`
- `backend/app/services/feature_svc.py`
- `backend/tests/test_feature_svc.py`
- `docs/model_architecture.md`
- `docs/project_journal.md`
- `docs/training_hyperparameters.md`

작업 전 `git status --short`에서 위 6개 파일이 이미 modified 상태였다. 이 감사 작업에서는 해당 파일들을 수정하지 않았다. 본 작업에서 추가한 파일은 요청 산출물인 `docs/cp29_system_risk_code_review_report.md`뿐이다.

주의: `.gitignore:4`의 `docs/cp*` 규칙 때문에 이 보고서 파일도 기본 `git status --short`에는 표시되지 않는다.

## 7. 실행한 읽기 전용 명령 목록

아래 명령은 모두 읽기 전용으로 실행했다. `rg`는 이 환경에서 접근 거부되어 `Select-String`으로 대체했다.

```powershell
Get-ChildItem -Recurse 기반 전체 파일 검색 2회
rg -n "on_conflict=|UNIQUE|CREATE TABLE IF NOT EXISTS predictions|prediction_evaluations|lower_breach|upper_breach|normalized_band_width" ai\storage.py backend\db\schema.sql backend\db\ensure_runtime_schema.py backend\app\repositories\ai_repo.py backend\app\repositories\prediction_repo.py backend\app\routers\v1\ai.py
rg -n "def prepare_dataset_splits|build_lazy_sequence_dataset|build_sequence_dataset\(|_PREPARED_SPLITS_CACHE|resolve_data_fingerprint|_FEATURE_CONTRACT_VERSION|fetch_training_frames|fetch_feature_index_frame|build_dataset_plan|save_registry|dropna|forward|ffill|asof|filing|release" ai\preprocessing.py
rg -n "status|failed_nan|save_model_run|save_predictions|save_prediction_evaluations|evaluate_bundle|early_stopping|checkpoint_selector|best_metrics|coverage_gate|grad_clip|compile_model|scheduler|run_training|save_checkpoint|run_epoch" ai\train.py
Select-String -Path ai\storage.py,backend\db\schema.sql,backend\app\repositories\ai_repo.py,backend\app\repositories\prediction_repo.py,backend\app\routers\v1\ai.py -Pattern "on_conflict=|UNIQUE|CREATE TABLE IF NOT EXISTS predictions|prediction_evaluations|lower_breach|upper_breach|normalized_band_width"
Select-String -Path ai\preprocessing.py -Pattern "def prepare_dataset_splits|build_lazy_sequence_dataset|build_sequence_dataset\(|_PREPARED_SPLITS_CACHE|resolve_data_fingerprint|_FEATURE_CONTRACT_VERSION|fetch_training_frames|fetch_feature_index_frame|build_dataset_plan|save_registry|dropna|forward|ffill|asof|filing|release"
Select-String -Path ai\train.py -Pattern "status|failed_nan|save_model_run|save_predictions|save_prediction_evaluations|evaluate_bundle|early_stopping|checkpoint_selector|best_metrics|coverage_gate|grad_clip|compile_model|scheduler|run_training|save_checkpoint|run_epoch"
Select-String -Path backend\app\services\api_service.py,backend\app\repositories\ai_repo.py,backend\app\repositories\prediction_repo.py,backend\app\routers\v1\ai.py -Pattern "fetch_latest_prediction|fetch_prediction_by_run|run_id|status|completed|fetch_model_runs|EVALUATION_COLUMNS|BACKTEST_COLUMNS|fetch_run_evaluations|fetch_run_backtests"
Select-String -Path ai\backtest.py,ai\evaluation.py,ai\inference.py,ai\preprocessing.py -Pattern "anchor_close|close|adjusted_close|prev_position|shift\(|strategy_return|fee|groupby|asof_date|spearman|top_k|long_short|raw_future_returns|actual_return"
Get-Content -Path ai\storage.py | Select-Object -First 260
Get-Content -Path ai\preprocessing.py | Select-Object -Skip 900 -First 260
Get-Content -Path ai\train.py | Select-Object -Skip 900 -First 360
Get-Content -Path backend\app\services\api_service.py | Select-Object -First 260
Get-Content -Path backend\app\routers\v1\ai.py | Select-Object -First 260
Get-Content -Path ai\train.py | Select-Object -Skip 1248 -First 280
Get-Content -Path ai\preprocessing.py | Select-Object -Skip 1120 -First 130
Get-ChildItem -Path backend -Recurse -Filter ensure_runtime_schema.py
Get-Content -Path backend\app\repositories\prediction_repo.py | Select-Object -First 120
Get-Content -Path backend\app\repositories\ai_repo.py | Select-Object -First 140
Select-String -Path backend\db\schema.sql,backend\db\scripts\ensure_runtime_schema.py -Pattern "CREATE TABLE IF NOT EXISTS model_runs|status|failed_nan|CREATE TABLE IF NOT EXISTS predictions|UNIQUE|prediction_evaluations|lower_breach|upper_breach|normalized_band_width|backtest_results|meta"
Get-Content -Path backend\db\schema.sql | Select-Object -Skip 170 -First 110
Get-Content -Path backend\db\scripts\ensure_runtime_schema.py | Select-Object -Skip 70 -First 110
Get-Content -Path ai\inference.py | Select-Object -Skip 240 -First 100
Get-Content -Path ai\models\patchtst.py | Select-Object -First 220
Get-Content -Path ai\models\revin.py | Select-Object -First 180
Get-Content -Path ai\models\common.py | Select-Object -First 220
Get-Content -Path ai\models\tide.py | Select-Object -First 240
Select-String -Path ai\models\patchtst.py,ai\models\revin.py,ai\models\common.py,ai\loss.py -Pattern "denormalize_target|target_channel_idx|ci_aggregate|ci_target_fast|band_mode|positive_width|softplus|lower_band|upper_band|WidthPenalty|upper - lower|clamp"
Get-Content -Path ai\loss.py | Select-Object -First 180
Get-Content -Path ai\ticker_registry.py | Select-Object -First 220
Select-String -Path ai\inference.py,ai\preprocessing.py,ai\ticker_registry.py,ai\train.py -Pattern "ticker_registry|load_registry|save_registry|build_registry|lookup_id|num_tickers|prepare_dataset_splits|resolve_bundle|config.ticker_registry_path|ticker_id"
Get-Content -Path backend\collector\sources\eodhd_prices.py | Select-Object -First 180
Get-Content -Path backend\collector\jobs\sync_prices.py | Select-Object -First 280
Get-Content -Path backend\collector\jobs\compute_indicators.py | Select-Object -First 240
Get-Content -Path backend\collector\jobs\sync_fundamentals.py | Select-Object -First 220
Select-String -Path backend\collector\sources\eodhd_prices.py,backend\collector\jobs\sync_prices.py,backend\collector\jobs\compute_indicators.py,backend\collector\jobs\sync_fundamentals.py,backend\collector\config.py -Pattern "except Exception|return pd.DataFrame|allow_yahoo_fallback|adjusted_close|source_empty|quota_hit|status|success|failed|empty|fallback|SourceLimitReached|upsert_records|filing_date|resolve_target_tickers|DEFAULT_TICKERS"
Get-Content -Path backend\collector\config.py | Select-Object -First 120
Get-ChildItem -Path backend\collector\pipelines -File
Get-Content -Path backend\collector\pipelines\daily_sync.py | Select-Object -First 260
Get-Content -Path backend\collector\pipelines\backfill_status.py | Select-Object -First 220
Get-Content -Path backend\collector\pipelines\bootstrap_backfill.py | Select-Object -First 260
Get-Content -Path backend\collector\pipelines\daily_market_sync.py | Select-Object -First 260
Get-Content -Path backend\collector\pipelines\preflight.py | Select-Object -First 220
Get-Content -Path backend\collector\repositories\base.py | Select-Object -First 260
Select-String -Path backend\app\services\feature_svc.py -Pattern "def build_features|resample|adjusted_close|target_close|merge_asof|filing_date|direction=|forward|ffill|bfill|has_macro|has_breadth|has_fundamentals|dropna|replace|inf|NaN|timeframe|1W|1M|macro|breadth|fundamentals"
Get-Content -Path backend\app\services\feature_svc.py | Select-Object -First 260
Get-Content -Path backend\app\services\feature_svc.py | Select-Object -Skip 260 -First 260
Get-Content -Path backend\app\repositories\market_repo.py | Select-Object -First 180
Get-Content -Path backend\collector\jobs\sync_stock_info.py | Select-Object -First 260
Get-ChildItem -Path backend\app\routers -Recurse -File
Select-String -Path backend\app\routers\*.py,backend\app\routers\v1\*.py -Pattern "get_stocks|stocks|price|prediction|latest|@router"
Get-Content -Path backend\app\main.py | Select-Object -First 140
Get-Content -Path backend\app\core\exceptions.py | Select-Object -First 180
Get-Content -Path backend\app\routers\v1\stocks.py | Select-Object -First 140
Get-Content -Path frontend\src\api\client.ts | Select-Object -First 180
Get-Content -Path scripts\start_demo.ps1 | Select-Object -First 260
Get-Content -Path scripts\check_demo_readiness.ps1 | Select-Object -First 300
Get-Content -Path frontend\package.json | Select-Object -First 160
Select-String -Path frontend\src\**\*.ts,frontend\src\**\*.tsx,scripts\start_demo.ps1,scripts\check_demo_readiness.ps1 -Pattern "NEXT_PUBLIC_BACKEND_URL|localhost|127.0.0.1|empty|fallback|fetch\(|api/v1|predictions|stocks|prices|health|Remove-Item|\.next|Start-Process|Cache-Control"
Get-Content -Path frontend\src\components\StockView.tsx | Select-Object -Skip 190 -First 120
Get-Content -Path frontend\src\components\BacktestView.tsx | Select-Object -First 140
Get-Content -Path frontend\src\components\TrainingView.tsx | Select-Object -First 120
Select-String -Path ai\preprocessing.py,frontend\src\api\client.ts,frontend\src\components\BacktestView.tsx -Pattern "SUPPORTED_AI_TIMEFRAMES|SUPPORTED_TIMEFRAMES|PredictionTimeframe|DisplayTimeframe|1M|isPredictionTimeframeEnabled|TIMEFRAMES"
Select-String -Path backend\requirements.txt,requirements.txt,.gitignore -Pattern "torch|statsmodels|docs/cp|\.env|\.pt|cache|parquet|artifacts|checkpoints|node_modules|\.next"
Get-Content -Path .gitignore | Select-Object -First 120
Get-Content -Path backend\requirements.txt | Select-Object -First 80
Get-Content -Path requirements.txt | Select-Object -First 80
git status --short
git ls-files .env .env.local "*.env"
git ls-files "*.pt" "*.pth" "*.onnx" "*.parquet" "ai/cache/*" "ai/artifacts/*" "backend/data/parquet/*" "docs/cp*"
Get-ChildItem -Path ai\cache,ai\artifacts -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 50 FullName,Length
Get-ChildItem -Path docs -Filter cp* -File -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime
git check-ignore -v docs\cp29_system_risk_code_review_report.md
Test-Path docs\cp29_system_risk_code_review_report.md
Get-Content -Path docs\cp29_system_risk_code_review_report.md | Select-Object -First 40
Get-Content -Path ai\sweep.py | Select-Object -First 300
Select-String -Path ai\sweep.py -Pattern "limit-tickers|compile|TrialPruned|except Exception|best_trial|best_value|prepare_dataset_splits|precomputed_bundles|study.optimize|gc_after_trial|n_trials"
Get-Content -Path ai\splits.py | Select-Object -First 260
Get-Content -Path ai\targets.py | Select-Object -First 220
Select-String -Path docs\model_architecture.md,docs\training_hyperparameters.md,docs\project_journal.md,README.md -Pattern "RevIN|PatchTST|TiDE|seq_len|252|104|gradient|scheduler|early|checkpoint|coverage_gate|feature_version|adjusted|publication|release|FRED|as-of|ticker_registry|lazy|cache|fingerprint|1M|월봉"
Get-Content -Path backend\app\schemas\ai.py | Select-Object -First 220
Get-Content -Path backend\app\schemas\stocks.py | Select-Object -First 180
Select-String -Path ai\train.py,README.md -Pattern "--seq-len|seq_len|seq-len|batch-size|epochs 50"
Get-ChildItem -Path ai\cache -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
Get-ChildItem -Path ai\artifacts -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
Get-ChildItem -Path backend\data\parquet -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
Get-ChildItem -Path docs -Filter cp* -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
```
