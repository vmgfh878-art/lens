# CP73-DG Lens 구조 최적화/보안 통합 read-only 감사

작성일: 2026-05-01  
범위: 코드/문서/로컬 파일 상태 기반 읽기 전용 감사  
주의: 모델 학습, 추론, DB 쓰기, 캐시 삭제/재생성, 서버 재시작, 포트 종료, npm 실행, save-run, W&B run 생성은 수행하지 않았다. `.env`는 키 이름만 확인했고 원문 값은 노출하지 않았다.

## 1. Executive Summary

현재 Lens의 가장 큰 구조 리스크는 모델 자체보다 `데이터 스냅샷을 식별하는 방식`, `line/band layer를 제품/API가 분리해서 다루는 준비도`, `대형 full run의 CPU/DataLoader 병목`, `run 선택/조회 쿼리 계약`에 있다.

P0로 확정할 즉시 보안 유출은 발견하지 못했다. 다만 `source_data_hash`가 실제 indicator 값 전체가 아니라 max date/count 중심으로 계산되어, 같은 row count와 latest date를 유지한 indicator 재계산은 stale cache를 감지하지 못할 수 있다. 이는 현재 full run 오염 여부가 아니라 미래 full run 재현성 붕괴 후보이며 P0 후보로 분류한다.

현재 진행 중인 1D LM/BM full run을 바로 오염시킨다고 확인된 증거는 없다. 단, CP70 문서 기준 source hash가 CP67 `f7c7b101`에서 `3ac43945`로 바뀌었고 feature_version은 `v3_adjusted_ohlc`로 유지된다. 즉, feature contract 변경 없이 데이터 해시만 바뀌는 정상 케이스가 이미 존재하며, 이 변경 이력을 명시적으로 남기는 manifest가 부족하다.

제품/API 관점에서는 line과 band를 하나의 prediction row로 소비하는 화면/차트 구조가 아직 남아 있다. `StockView`는 run 후보를 `patchtst` 중심으로 찾고, `Chart`는 단일 `PredictionResult` 안의 `line_series/lower/upper`를 같이 그린다. BM full run 결과를 별도 AI band layer로 노출하려면 run 선택, provenance, overlay bundle 계약 정리가 먼저 필요하다.

학습 성능 관점에서는 Windows에서 `num_workers=auto`가 0으로 떨어지고, lazy `SequenceDataset.__getitem__`이 타깃 배열을 매 샘플 구성한다. GPU VRAM 여유가 있어도 CPU/DataLoader가 병목일 가능성이 높다. CNN-LSTM은 CUDA에서 cuDNN을 명시적으로 끄는 안정성 우회가 있어 BM full run 속도 병목 후보지만, 현재 full run 중에는 건드리면 안 된다.

## 2. P0/P1/P2/P3 리스크 표

| 등급 | 리스크 | 근거 | 영향 | 권장 조치 |
|---|---|---|---|---|
| P0 후보 | indicator 값이 바뀌어도 max date/count가 같으면 stale feature cache를 재사용할 수 있음 | `ai/preprocessing.py:626` `resolve_data_fingerprint()`가 timeframe, price max date, indicator max date, indicator count 중심 payload만 사용. `ai/preprocessing.py:566`은 cache path가 있으면 즉시 로드 | 재현성 붕괴, stale 데이터로 full run 가능 | data fingerprint에 indicator/price content checksum 또는 per-ticker digest 추가 |
| P1 | legacy/composite 필터가 DB pagination 후 Python에서 적용됨 | `backend/app/repositories/ai_repo.py:49`, `:60`-`:71` | 최신 제품 run이 첫 페이지 밖으로 밀려 프론트가 못 찾을 수 있음 | legacy 제외 조건을 DB query 단계로 이동하거나 충분한 over-fetch 정책 추가 |
| P1 | run_id 없는 latest prediction 조회가 run 상태/legacy/role을 확인하지 않음 | `backend/app/repositories/prediction_repo.py:14`-`:33`, `backend/app/services/api_service.py:152`-`:162` | failed/legacy/non-product prediction이 latest로 노출될 위험 | latest prediction은 completed product run join 기준으로 조회 |
| P1 | 프론트가 LM/BM layer 분리 표시 준비가 부족함 | `frontend/src/components/StockView.tsx:79`, `frontend/src/components/Chart.tsx:60`, `frontend/src/components/TrainingView.tsx:61` | BM full run 저장 후 제품 화면에 band layer로 자연스럽게 표시하기 어려움 | line layer, band layer, overlay bundle API/프론트 계약 분리 |
| P1 | run/evaluation/backtest 조회용 인덱스 부족 | `backend/db/schema.sql`의 `model_runs`, `predictions`, `prediction_evaluations`, `backtest_results` index 구성 | run이 쌓일수록 latest/API 조회 지연 및 Supabase timeout | status/timeframe/model/created_at, run_id/ticker/asof index 추가 검토 |
| P1 | Windows DataLoader `num_workers=0` fallback과 lazy dataset target 구성 병목 | `ai/train.py:823`-`:880`, `ai/preprocessing.py:151`-`:200` | GPU 사용률 저하, full run 시간 증가 | Windows용 precomputed bundle/worker 전략 별도 실험 |
| P1 | CNN-LSTM CUDA 경로에서 cuDNN 비활성화 | `ai/models/cnn_lstm.py:97`-`:108` | BM full run 속도 저하 | full run 종료 후 안정성 재현 테스트와 선택적 cuDNN ablation |
| P1 | W&B 기본값이 켜져 있어 의도치 않은 run 생성 가능 | `ai/train.py:669`, `ai/train.py:1314`-`:1335`; 로컬 `wandb/run-20260501_105055-s902l1t2` 존재 | 실험 로그 혼선, 공개/비공개 정책 리스크 | full run 스크립트에서 `WANDB_MODE`, project/entity/privacy 정책 명시 |
| P2 | feature_version 변경 없이 source hash만 변경되는 정상 케이스의 manifest 부족 | CP70 문서상 current source hash `3ac43945`, CP67 hash `f7c7b101`, feature_version `v3_adjusted_ohlc` | 실험 비교 시 데이터 차이를 놓칠 수 있음 | run meta에 source hash, cache path, backfill marker, registry hash를 명시 |
| P2 | default ticker registry와 hash registry가 혼재 | `ai/ticker_registry.py:10`-`:32`, `ai/inference.py:113`-`:133` | checkpoint mapping mismatch 위험은 줄었지만 운영 혼동 가능 | registry provenance를 run meta와 cache manifest에 통일 |
| P2 | stock search fallback이 대량 테이블에서 client-side filter 가능 | `backend/app/repositories/market_repo.py:163`-`:188` | ticker 수 증가 시 API 지연 | DB 검색 인덱스/검색 query 정리 |
| P2 | Render daily sync가 전체 target ticker 가격 sync를 기본으로 잡을 수 있음 | `backend/collector/pipelines/daily_market_sync.py:134`-`:146`, `render.yaml` | cron timeout, derived 계산 미실행 | batch/env limit 정책과 실패 후 재시도 전략 문서화 |
| P2 | collector logging에 redaction 계층 없음 | `backend/collector/utils/logging.py` | 향후 caller가 token/DSN을 넘기면 로그 노출 가능 | key/token/password/url 필드 redaction helper 추가 |
| P2 | CORS는 origin 제한이 있으나 methods/headers wildcard | `backend/app/main.py:19`-`:29` | 운영 origin 관리 실수 시 surface 확대 | 운영 env origin allowlist와 배포 체크 추가 |
| P3 | 오래된 demo/readiness 스크립트가 patchtst 중심 | `scripts/check_demo_readiness.ps1`, `scripts/start_demo.ps1` | 발표 절차 혼동 | LM/BM layer 기준 readiness v2 작성 |
| P3 | 일부 collector 파일에 mojibake 주석 존재 | `backend/collector/sources/eodhd_prices.py` | 유지보수 가독성 저하 | 별도 정리 CP에서 인코딩 정리 |

## 3. 현재 full run 결과를 오염시킬 수 있는 위험

확정 오염 증거는 발견하지 못했다. 현재 실행 중인 Python 프로세스는 확인했으나 명령줄 조회는 권한 문제로 실패했고, 프로세스 종료나 간섭은 하지 않았다. GPU는 read-only로 `NVIDIA GeForce RTX 5060 Ti`, 사용 메모리 약 878MiB, 총 16311MiB 스냅샷만 확인했다. compute app 상세는 권한 부족으로 신뢰 가능한 command attribution을 얻지 못했다.

오염 가능성이 가장 큰 구조 지점은 cache fingerprint다. `ai/preprocessing.py:626`의 `resolve_data_fingerprint()`는 `indicator_max_date`, `indicator_count`, `price_max_date`를 반영하지만 indicator 값 자체, per-ticker row count, OHLC/ratio 값 checksum은 반영하지 않는다. 그래서 CP64 같은 full backfill 이후 max date/count가 바뀐 경우에는 hash가 바뀌지만, 같은 날짜/row 수에서 값을 재계산하는 repair는 cache path를 바꾸지 못할 수 있다.

CP70 문서 기준으로 CP67 hash `f7c7b101`과 current hash `3ac43945`가 달라졌고, feature_version은 `v3_adjusted_ohlc`로 유지된다. 이는 현재 run이 어느 hash/cache를 사용했는지가 실험 결과 해석에 중요하다는 뜻이다. full run 결과를 받을 때는 반드시 run config 또는 로그에서 `source_data_hash`, `feature_version`, `cache path`, `ticker registry path`를 확인해야 한다.

## 4. 데이터/cache/hash/registry 감사

`ai/cache` 로컬 현황은 `.pt` 72개, `.json` 7개, `.pt` 총 약 6.08GB다. 그중 `features_1D` 33개가 약 4.94GB, `feature_index_1D` 33개가 약 0.84GB로 대부분을 차지한다. 1W는 `features_1W` 3개, `feature_index_1W` 3개이며 1M feature cache는 확인되지 않았다.

cache path는 `ai/preprocessing.py:680`-`:714`에서 timeframe, source columns, calendar columns, feature contract version, tickers/limit_tickers, data hash를 섞어 만든다. 이 구조는 ticker subset별 캐시 분리를 지원하지만, data hash가 약하면 stale data를 구분하지 못한다.

`_maybe_warn_stale_cache()`는 `ai/preprocessing.py:671`-`:677`에서 target cache path가 없고 같은 prefix의 다른 파일이 있을 때만 경고한다. 이미 존재하는 target path가 stale인지 확인하지는 않는다. 따라서 stale cache 감지는 “다른 이름의 캐시가 보인다” 수준이고, manifest 기반 검증은 아니다.

feature index cache는 `ai/preprocessing.py:461`-`:486`에서 `indicators`의 `ticker,timeframe,date`만 가져와 eligibility/split 계획에 사용한다. 이 경로는 비교적 가볍지만, indicator full backfill 후에도 값 변화는 index cache에 반영되지 않는다. 즉, eligibility는 맞아도 feature tensor 값은 별도 cache fingerprint에 의존한다.

ticker registry는 두 계열이 혼재한다. `ai/ticker_registry.py:10`-`:12`는 timeframe별 default path를 쓰고, `ai/ticker_registry.py:23`-`:32`는 eligible ticker hash 기반 path를 만든다. inference는 `ai/inference.py:113`-`:133`에서 checkpoint registry path를 강제하고 timeframe/num_tickers mismatch를 막는다. 이 점은 CP30 이후 개선된 부분이다. 다만 training/cache/reporting에서 default registry와 hash registry의 관계가 명확히 기록되지 않으면 실험 재현성이 흔들릴 수 있다.

## 5. 학습 준비 및 GPU/CPU 병목 후보

`ai/sweep.py:240`-`:267`과 CP66/CP67 계열 스크립트는 precomputed bundle을 한 번 만들어 여러 trial에 재사용하는 구조를 이미 갖고 있다. 이는 좋은 방향이다. 반면 `ai/train.py:1451`-`:1520`의 일반 `run_training()`은 `precomputed_bundles`가 없으면 매 run마다 `prepare_dataset_splits()`를 호출한다. full/50/100 실험을 반복할 때 같은 split/feature를 재사용하지 않으면 불필요한 준비 시간이 누적된다.

`ai/preprocessing.py:1194`-`:1265`의 `_PREPARED_SPLITS_CACHE`는 프로세스 내 in-memory cache다. 프로세스가 바뀌면 재사용되지 않는다. 장기적으로는 disk-backed split manifest와 bundle manifest가 필요하다.

lazy `SequenceDataset`은 메모리 절약에는 유리하지만, `ai/preprocessing.py:151`-`:200`에서 `__getitem__`마다 window slice와 target 배열을 구성한다. Windows에서는 `ai/train.py:823`-`:834`의 `resolve_num_workers("auto")`가 0을 반환한다. 이 조합은 GPU가 기다리는 CPU/DataLoader 병목 후보가 된다.

`ai/train.py:837`-`:880`의 DataLoader는 CUDA일 때 pin_memory를 켜지만, worker가 0이면 persistent worker/prefetch 이점이 없다. Windows에서 worker를 무작정 늘리는 것은 spawn 비용과 pickle 제약이 있어 위험하지만, full run 종료 후 `precomputed TensorDataset`, `num_workers`, `persistent_workers`, `prefetch_factor`를 작게 ablation하는 가치가 있다.

PatchTST는 VRAM 여유가 있으면 batch를 더 키울 여지가 있을 수 있다. 단 이번 감사에서는 새 실행을 하지 않았고, 현재 GPU snapshot만으로 batch 상향을 결론낼 수 없다. epoch 로그의 `vram_peak_allocated_mb`, `epoch_seconds`를 받은 뒤 판단해야 한다.

CNN-LSTM은 `ai/models/cnn_lstm.py:97`-`:108`에서 CUDA tensor일 때 cuDNN을 끈다. 주석상 Windows native CUDA crash 회피 목적이다. 속도 병목 후보지만 안정성 우회로 보이므로 현재 full run 중에는 변경 대상이 아니다. full run 이후 별도 재현 테스트로만 판단해야 한다.

## 6. W&B/logging/reproducibility 리스크

`ai/train.py:669`는 CLI 기본 `--wandb`가 true다. `ai/train.py:1314`-`:1335`의 `maybe_init_wandb()`는 `WANDB_MODE=disabled/offline`이 아니면 W&B run을 만든다. 로컬 `wandb/run-20260501_105055-s902l1t2`가 존재해 최근/현재 run이 W&B를 사용했을 가능성이 있다. 이번 감사에서는 W&B를 실행하거나 새 run을 만들지 않았다.

run name은 기본적으로 `"{model}-{timeframe}-{run_id[:8]}"`다. CP49 subprocess 계열은 `ai/cp49_patchtst_horizon_rescue.py:123`-`:170`에서 candidate별 의미 있는 W&B name을 넘기지 않는다. 기본 이름은 유니크하지만, 실험 후보 비교에는 약하다.

`ai/train.py:1680`-`:1720`은 epoch마다 `epoch_seconds`, `elapsed_seconds`, `estimated_remaining_seconds`, `vram_peak_allocated_mb`를 JSON summary로 남긴다. 후속 비교에 필요한 기본 로그는 있다. 다만 DataLoader wait, CPU preprocessing time, cache load/build time, split build time이 분리되어 있지 않아 병목 attribution에는 부족하다.

`ai/train.py:1853`-`:1898`은 save-run 시 `feature_version=FEATURE_CONTRACT_VERSION`, `band_mode`, config, status를 저장한다. 이는 CP30 이후 개선된 부분이다. 다만 source hash, cache file path, ticker registry path/hash, precomputed bundle path가 run meta에 항상 충분히 들어가는지 full run 결과에서 확인이 필요하다.

## 7. 보안 리스크

`.env`는 존재 키 이름만 확인했다. 확인된 키 이름에는 `DB_PASSWORD`, `SUPABASE_KEY`, `EODHD_API_KEY`, `FMP_API_KEY`, `FRED_API_KEY`, `WANDB_API_KEY` 등이 있다. 원문 값은 출력하거나 보고서에 기록하지 않았다.

`.gitignore`는 `.env`, `.env.local`, `*.env`, `.venv`, `node_modules`, `.next`, `*.pt`, `ai/cache/ticker_id_map_*.json`, `wandb/`, log/artifact 계열을 무시한다. 큰 보안 구멍은 발견하지 못했다.

프론트는 `frontend/src/api/client.ts`에서 `NEXT_PUBLIC_BACKEND_URL`만 사용한다. `frontend/.env.local.example`에도 public backend URL만 있어 현재 확인 범위에서는 secret이 bundle에 들어갈 구조는 보이지 않았다.

`backend/app/main.py:19`-`:29`의 CORS는 `BACKEND_CORS_ORIGINS` env를 사용하고 기본값은 localhost 두 개다. wildcard origin은 기본이 아니다. 다만 `allow_methods=["*"]`, `allow_headers=["*"]`는 운영 origin allowlist 관리가 실패할 경우 surface를 넓힌다.

`backend/collector/utils/logging.py`는 전달받은 fields를 그대로 JSON으로 찍는다. 현재 검색 범위에서 API key를 직접 넘기는 호출은 확인하지 못했지만, redaction 계층이 없으므로 향후 token/password/db url 필드가 들어가면 그대로 노출될 수 있다.

`ai/preprocessing.py:300`-`:317`은 DB password를 포함한 DSN string을 만들고, `ai/preprocessing.py:324`의 `_ENGINE_CACHE` 키로 사용한다. 출력하지 않는 한 문제가 되지는 않지만, 예외/debug repr에 dict key가 노출되면 secret leak이 될 수 있다. DSN 자체를 로그에 찍지 않는 정책이 필요하다.

`backend/requirements.txt`에는 과거 torch CPU wheel overwrite를 막기 위한 주석이 있어 재발 위험을 낮춘 상태다. 다만 dependency vulnerability audit은 네트워크/빌드 금지 범위라 수행하지 않았다.

## 8. API/run 선택 계약 리스크

`/api/v1/ai/runs`는 `backend/app/routers/v1/ai.py:178`-`:198`에서 기본 `status=completed`, `model_name=patchtst`, `include_legacy=false`로 조회한다. repository에서도 legacy를 필터하지만 `backend/app/repositories/ai_repo.py:49`-`:71`처럼 Supabase range를 먼저 적용한 뒤 Python에서 제외한다. legacy/composite run이 앞쪽에 많으면 실제 제품 run이 결과에서 빠질 수 있다.

`include_legacy=true` 경로는 demo readiness에서 사용된다. 이는 과거 composite demo 확인에는 유용하지만, CP58 이후 line/band layer 분리 의도와 혼동될 수 있다. legacy 조회는 명시적인 진단/이관 도구로만 남기는 것이 안전하다.

run_id가 없는 prediction 최신 조회는 `backend/app/repositories/prediction_repo.py:14`-`:33`에서 `predictions`만 보고 최신 asof/decision_time을 고른다. `model_runs.status`, legacy 여부, role, layer provenance를 join하지 않는다. `backend/app/services/api_service.py:152`-`:162`도 run_id가 없으면 model_run을 검증하지 않는다. 제품 기본 조회는 run_id 기반 또는 product-eligible run join 기반이어야 한다.

evaluation/backtest API는 run_id 기반 조회가 있고 기본적인 limit은 있다. 그러나 `backend/app/routers/v1/ai.py:217`-`:246`에는 offset이 없다. 대량 결과가 쌓일 경우 페이지네이션과 index가 같이 필요하다.

## 9. DB/index/query 성능 리스크

`backend/db/schema.sql` 기준 `price_data`에는 `(ticker,date desc)`, `indicators`에는 `(ticker,timeframe,date desc)` index가 있어 차트 기본 조회에는 맞다.

`model_runs`는 primary key 외에 `status,timeframe,model_name,created_at desc` 조합 index가 보이지 않는다. `/ai/runs`가 status/timeframe/model_name/created_at 정렬로 쓰이므로 run이 많아지면 latest 조회가 느려질 수 있다.

`predictions`는 unique `(run_id,ticker,model_name,timeframe,horizon,asof_date)`와 `(ticker,timeframe,asof_date desc,decision_time desc)` index가 있다. run_id 기반 조회 `fetch_prediction_by_run()`에는 `(run_id,ticker,asof_date desc,decision_time desc)` 또는 `(ticker,run_id,asof_date desc,decision_time desc)` index가 더 직접적이다.

`prediction_evaluations`는 unique `(run_id,ticker,timeframe,asof_date)`가 있지만 API 조회는 run_id/ticker/timeframe/asof 정렬이다. unique index가 일부 도움은 되지만, desc order와 limit 패턴에 맞춘 index가 있으면 더 안전하다.

`backtest_results`는 unique `(run_id,strategy_name,timeframe)`가 있고 `(strategy_name,timeframe,created_at desc)` index가 있다. API는 run_id를 먼저 필터하므로 `(run_id,strategy_name,timeframe,created_at desc)`가 더 맞다.

`predictions.meta` JSONB에는 line/band provenance와 composition meta가 들어갈 수 있다. JSONB 조건 검색이 제품 경로에 들어가면 별도 generated column 또는 GIN index가 필요하다. 현재는 구조 위험으로만 확인했다.

## 10. 프론트 layer 표시 계약 리스크

`frontend/src/components/StockView.tsx:79`-`:80`은 AI run 후보를 `patchtst`로 제한한다. `frontend/src/components/TrainingView.tsx:61`-`:107`과 `frontend/src/components/BacktestView.tsx:8`-`:10`도 patchtst 중심이다. BM full run이 CNN-LSTM/TiDE 계열로 저장되면 제품 화면에서 자동으로 band layer 후보가 되지 않는다.

`frontend/src/components/Chart.tsx:60`-`:123`은 단일 `PredictionResult`에서 `line_series`, `upper`, `lower`를 함께 읽어 overlay state를 만든다. `frontend/src/components/Chart.tsx:291`-`:321`도 하나의 prediction object에서 line과 band를 같이 그린다. CP58의 의도인 “AI line과 AI band는 별도 보조지표 layer”와는 아직 완전히 맞지 않는다.

`frontend/src/components/StockView.tsx:407`-`:423`은 forecast_dates와 line/upper/lower 길이가 모두 맞아야 overlay를 표시한다. line-only 또는 band-only layer를 독립적으로 표시하는 계약이 아니다.

`frontend/src/components/StockView.tsx:323`-`:329`와 `frontend/src/components/Chart.tsx:65`-`:75`는 1M price-only 정책을 유지한다. 이 부분은 의도와 맞다.

`frontend/src/components/IndicatorPanel.tsx:51`-`:63`은 chart visible dates에 맞춰 보조지표를 필터링하고, `:97`-`:116`에서 timeline date 기준으로 path를 만든다. 날짜축 정렬은 큰 방향이 맞다. 다만 forecast whitespace dates가 붙은 chart timeline에서는 미래 구간 indicator가 비어 보일 수 있으므로 empty state 문구가 필요할 수 있다.

## 11. 운영/배포/크론잡 리스크

`render.yaml`의 `lens-daily-market-sync`는 `python -m backend.collector.pipelines.daily_market_sync --indicator-lookback-days 60`을 실행한다. `backend/collector/pipelines/daily_market_sync.py:43`의 CLI 기본값은 14지만 Render는 60으로 덮는다. ATR 14 기준 1D 최근 계산에는 충분할 가능성이 높다. 1W/1M 장기 backfill 목적에는 별도 full backfill 경로가 필요하다.

`daily_market_sync.py:134`-`:146`은 target tickers 전체를 price sync 대상으로 잡을 수 있다. batch limit env가 없으면 timeout 위험이 있다. `daily_market_sync.py:157`-`:169`는 coverage가 부족하면 derived 계산 전에 실패 처리한다. 가격 sync 실패가 indicators/sector/breadth 업데이트까지 막을 수 있다.

`scripts/start_demo.ps1`은 Windows PowerShell 기반으로 backend/frontend hidden window를 띄우고 `npm run dev`를 실행한다. 발표 데모에는 편하지만 Windows 전용 의존성이 강하다. 이번 CP에서는 실행하지 않았다.

`scripts/check_demo_readiness.ps1`은 completed patchtst run과 usable prediction을 찾고 legacy composite도 별도 확인한다. 현 demo readiness는 “PatchTST latest forecast demo”에는 맞지만 “line layer + band layer 분리 제품” readiness와는 다르다.

`.next`는 gitignore에 포함되어 있어 깨짐 재발 시 빌드 산출물 자체는 추적되지 않는다. dev/prod 실행 방식 분리는 문서와 script 수준에서 더 명확히 해야 한다.

## 12. 지금 수정해야 할 것

현재 full run 중에는 코드 수정 금지 원칙을 유지해야 한다. 따라서 “지금”은 실행 중인 run을 건드리지 않고 결과 해석에 필요한 로그/메타 확인 항목만 준비하는 것이 맞다.

1. full run 결과 수령 시 `source_data_hash`, `feature_version`, `cache path`, `ticker registry path`, `eligible ticker count`, `MODEL_N_FEATURES`, `atr_ratio in model features=false`를 체크리스트로 확인한다.
2. W&B run이 생성되어 있다면 공개/비공개 설정과 run name, config의 hash/registry 기록 여부를 확인한다. secret 값은 확인하거나 출력하지 않는다.
3. 현재 full run이 실패하더라도 프로세스 종료/재시작 없이 종료 로그를 먼저 확보한다.
4. 제품 데모에는 BM run이 자동 노출되지 않을 수 있음을 미리 표시한다. 현재 프론트는 patchtst 중심이며 layer 분리 구조가 아니다.

## 13. full run 결과 받은 뒤 수정해야 할 것

1. `resolve_data_fingerprint()`를 content-aware fingerprint로 강화한다. 최소한 indicator/price per-timeframe row count, min/max date, per-ticker count, 핵심 ratio/atr 컬럼 checksum, feature contract version, backfill marker를 manifest에 남긴다.
2. feature cache manifest를 도입해 `.pt` 파일 옆에 source hash, feature columns, data snapshot, registry hash, 생성 명령, 생성 시각을 저장한다.
3. `/api/v1/ai/runs` legacy 필터를 DB query 단계로 이동하거나 over-fetch 후 limit을 다시 적용한다.
4. run_id 없는 latest prediction 조회를 product-eligible completed run join 기반으로 바꾼다.
5. line layer, band layer, overlay bundle을 API와 프론트에서 분리한다. Chart는 line-only, band-only, combined overlay를 각각 받을 수 있어야 한다.
6. DB index를 run selection과 run_id 조회 패턴에 맞춰 보강한다.
7. Windows full run 병목을 측정한다. cache load, split build, DataLoader wait, forward/backward time을 분리 로깅하고, precomputed TensorDataset/num_workers/prefetch/batch size를 작은 smoke에서만 ablation한다.
8. CNN-LSTM cuDNN off 정책은 Windows crash 재현 테스트 후 선택적으로 완화한다.
9. W&B name/group/tags 정책을 CP 단위로 고정하고 `save-run`/W&B 생성 정책을 스크립트마다 명시한다.
10. collector logging에 secret redaction helper를 추가한다.

## 14. 읽기 전용 명령 목록

아래 명령은 읽기 전용 확인 목적으로만 실행했다.

```powershell
Get-Command rg -ErrorAction SilentlyContinue
git status --short
Get-Process python,pythonw,node -ErrorAction SilentlyContinue | Select-Object ProcessName,Id,CPU,WorkingSet,StartTime,Path
Get-CimInstance Win32_Process -Filter "ProcessId=35188 or ProcessId=40596" | Select-Object ProcessId,CommandLine
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
git ls-files
Get-ChildItem -Path ai/cache -File -Recurse | Group-Object Extension
Get-ChildItem -Path ai/cache -File -Recurse | Sort-Object Length -Descending | Select-Object -First 20 FullName,Length,LastWriteTime
Get-ChildItem -Path ai/artifacts/checkpoints -File -Recurse | Sort-Object Length -Descending | Select-Object -First 20 FullName,Length,LastWriteTime
Get-ChildItem -Path wandb -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 10 FullName,LastWriteTime
Select-String -Path .env -Pattern '^[A-Za-z_][A-Za-z0-9_]*='
Select-String -Path ai/preprocessing.py -Pattern '_FEATURE_CONTRACT_VERSION|MODEL_FEATURE_COLUMNS|resolve_data_fingerprint|resolve_feature_cache_path|fetch_training_frames|fetch_feature_index_frame|SequenceDataset|prepare_dataset_splits'
Select-String -Path ai/train.py -Pattern 'wandb|num_workers|pin_memory|prefetch|compile|epoch_seconds|vram_peak|feature_version|save_model_run'
Select-String -Path ai/ticker_registry.py -Pattern 'default_registry_path|registry_path_for_tickers|build_and_save_registry|load_registry'
Select-String -Path ai/inference.py -Pattern 'resolve_checkpoint_ticker_registry|registry'
Select-String -Path ai/models/cnn_lstm.py -Pattern 'cudnn|autocast|fp32'
Select-String -Path backend/app/repositories/ai_repo.py -Pattern 'fetch_model_runs|include_legacy|legacy|evaluation|backtest'
Select-String -Path backend/app/repositories/prediction_repo.py -Pattern 'fetch_latest_prediction|fetch_prediction_by_run'
Select-String -Path backend/app/services/api_service.py -Pattern 'run_id|prediction|model_run'
Select-String -Path backend/app/repositories/market_repo.py -Pattern 'indicator|atr_ratio|search'
Select-String -Path backend/db/schema.sql -Pattern 'model_runs|predictions|prediction_evaluations|backtest_results|create index|unique'
Select-String -Path frontend/src/components/StockView.tsx -Pattern 'AI_RUN_MODEL_CANDIDATES|includeLegacy|latestRun|prediction|1M|composite'
Select-String -Path frontend/src/components/Chart.tsx -Pattern 'PredictionResult|upper|lower|line_series|1M|forecast'
Select-String -Path frontend/src/components/TrainingView.tsx -Pattern 'TRAINING_RUN_MODELS|composite|line|band'
Select-String -Path frontend/src/components/BacktestView.tsx -Pattern 'DEMO_RUN_MODELS|includeLegacy|timeframe'
Select-String -Path frontend/src/components/IndicatorPanel.tsx -Pattern 'timelineDates|visible|indicator'
Select-String -Path render.yaml -Pattern 'daily_market_sync|indicator-lookback-days|startCommand|buildCommand'
Select-String -Path backend/collector/pipelines/daily_market_sync.py -Pattern 'indicator_lookback_days|sync_prices|run_indicators|coverage|skip_derived'
Select-String -Path scripts/check_demo_readiness.ps1 -Pattern 'include_legacy|patchtst|prediction|completed'
Select-String -Path scripts/start_demo.ps1 -Pattern 'npm run dev|uvicorn|BACKEND_CORS_ORIGINS|NEXT_PUBLIC_BACKEND_URL'
```

`Get-CimInstance Win32_Process`는 권한 부족으로 command line 확인에 실패했다. 이 실패 때문에 우회적으로 프로세스를 건드리지는 않았다.
