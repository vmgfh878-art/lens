# CP129-UXD/ARCH-REVIEW 제품 임시 구조와 legacy 잔재 감사

## 1. 핵심 결론

이번 감사는 코드 수정, DB read/write, 모델 학습/inference, npm build, 서버 재시작 없이 정적 코드 읽기만으로 진행했다. 변경한 파일은 이 보고서뿐이다.

가장 위험한 잔재는 세 가지다. 첫째, 제품 차트와 백테스트 전략 신호가 새 제품용 rolling history endpoint가 아니라 기존 `predictions/history`를 계속 사용한다. 둘째, `market_breadth`와 `sector_returns` 계산이 `price_data`를 source/provider 없이 읽고 source 없는 테이블에 덮어쓴다. 셋째, 일반 `ai/inference.py --save` 경로가 제품 latest-only 저장 guard를 우회해 split history를 다시 Supabase에 저장할 수 있다.

DB 내용은 읽지 않았으므로 “현재 실제로 오염되어 있다”는 결론은 내리지 않는다. 다만 코드 구조상 old history row, test split row, mixed provider row가 제품 화면/피처에 섞일 수 있는 경로가 남아 있으므로 Phase 1 종료 전에는 막는 편이 안전하다.

## 2. P0/P1/P2 findings

| ID | 심각도 | 요약 | 근거 | 영향 |
|---|---:|---|---|---|
| CP129-P0-1 | P0 | 제품 차트/전략 신호가 제품 rolling history 대신 기존 prediction history를 사용한다 | `frontend/src/components/StockView.tsx:787`, `frontend/src/components/BacktestView.tsx:613`, `frontend/src/api/client.ts:231`, `backend/app/routers/v1/stocks.py:112`, `scripts/cp128_product_rolling_prediction_history_replay.py:540` | 기존 DB에 test split 또는 bulk history row가 남아 있으면 제품 history처럼 보일 수 있다. 반대로 latest-only row만 있으면 차트의 history가 비어 제품 기능이 깨질 수 있다. |
| CP129-P0-2 | P0 | `market_breadth`와 `sector_returns`가 source/provider 없이 `price_data`를 읽는다 | `backend/collector/jobs/compute_market_breadth.py:78`, `backend/collector/jobs/compute_market_breadth.py:163`, `backend/collector/jobs/sync_sector_returns.py:22` | yfinance/EODHD가 병렬 저장된 상태에서 breadth/sector 지표가 섞여 계산될 수 있다. 이 값은 indicator feature로 들어가므로 모델 입력 오염으로 이어질 수 있다. |
| CP129-P0-3 | P0 | 일반 inference 저장 경로가 latest-only 제품 저장 guard를 우회한다 | `ai/inference.py:66`, `ai/inference.py:67`, `ai/inference.py:478`, `ai/storage.py:23`, `ai/storage.py:95` | `--split train --save` 같은 실행으로 대량 prediction/evaluation row가 다시 Supabase에 저장될 수 있다. 제품 DB thin 정책과 제품 latest-only 계약을 동시에 깬다. |
| CP129-P1-1 | P1 | 제품 run_id가 프론트와 운영 스크립트에 하드코딩되어 있다 | `frontend/src/components/StockView.tsx:84`, `frontend/src/components/BacktestView.tsx:18`, `frontend/src/components/TrainingView.tsx:79`, `scripts/cp99_1d_product_loop_thin_upload.py:47`, `scripts/cp128_product_rolling_prediction_history_replay.py:20` | 제품 모델 교체, rollback, 1W 확장 시 배포 코드 수정 없이는 제품 슬롯과 화면이 맞지 않는다. run_id가 제품 계약이 아니라 구현 상수로 굳어져 있다. |
| CP129-P1-2 | P1 | run_id 기반 prediction 조회가 timeframe/horizon/role/product_latest_only까지 묶지 않는다 | `backend/app/repositories/prediction_repo.py:44`, `backend/app/repositories/prediction_repo.py:70`, `backend/app/services/api_service.py:182`, `backend/app/services/api_service.py:216` | run_id만 믿고 최신 row를 가져오므로 같은 run_id 아래 실험성 row, 다른 horizon, legacy row가 섞이면 방어가 약하다. 현재는 run_id 운영 규칙에 의존한다. |
| CP129-P1-3 | P1 | 1W는 client 타입상 prediction-enabled인데 화면에서는 준비 중으로 즉시 중단한다 | `frontend/src/api/client.ts:171`, `frontend/src/components/StockView.tsx:769`, `frontend/src/components/TrainingView.tsx:105`, `frontend/src/components/TrainingView.tsx:117` | 사용자와 검토자는 1W가 지원되는지 준비 중인지 헷갈릴 수 있다. 제품 계약상 1W 미출시라면 client 타입부터 비활성으로 맞춰야 한다. |
| CP129-P1-4 | P1 | model_runs의 checkpoint_path가 로컬 repo path에 강하게 묶여 있다 | `ai/train.py:1767`, `ai/train.py:2236`, `ai/train.py:2274`, `ai/train.py:2342`, `ai/inference.py:439` | 다른 작업 디렉터리, 배포 서버, artifact 이동 후 inference 재현성이 깨질 수 있다. 제품 run registry가 artifact 위치와 hash를 함께 관리하지 않는다. |
| CP129-P2-1 | P2 | 12개 scanner subset은 UI에 표시되지만 여전히 제품 로직 상수다 | `frontend/src/components/BacktestView.tsx:23`, `frontend/src/components/BacktestView.tsx:1006`, `frontend/src/components/BacktestView.tsx:1263` | 현재는 “12개 주요 종목”이라고 밝혀 치명도는 낮다. 다만 scanner API로 대체되지 않으면 제품 기능처럼 굳을 위험이 있다. |
| CP129-P2-2 | P2 | Chart rolling history 대표 horizon이 `HISTORY_HORIZON_INDEX = 4`로 고정되어 있다 | `frontend/src/components/Chart.tsx:49`, `frontend/src/components/Chart.tsx:223`, `scripts/cp128_product_rolling_prediction_history_replay.py:459` | 현재 CP128은 h5 scalar를 의도하므로 당장은 맞다. 하지만 horizon이 바뀌면 차트는 계약 변경을 모르고 5번째 값만 고른다. |
| CP129-P2-3 | P2 | TrainingView가 composite를 legacy filter로 걸러내지만 fetch 범위에는 포함한다 | `frontend/src/components/TrainingView.tsx:81`, `frontend/src/components/TrainingView.tsx:534`, `frontend/src/components/TrainingView.tsx:1743` | 지금은 `isLegacyRun`으로 숨긴다. 향후 legacy 표식이 누락된 composite-like run이 생기면 이전 실험이 제품 후보 목록에 재노출될 수 있다. |
| CP129-P2-4 | P2 | legacy snapshot bootstrap이 indicators source/provider를 EODHD로 고정한다 | `backend/collector/pipelines/bootstrap_snapshot.py:45`, `backend/collector/pipelines/bootstrap_snapshot.py:46`, `backend/collector/pipelines/bootstrap_snapshot.py:63` | yfinance 전환 후 이 스크립트를 재사용하면 provider와 실제 입력 파일이 어긋난 indicator row를 만들 수 있다. legacy 전용으로 격리해야 한다. |
| CP129-P2-5 | P2 | stock search fallback이 local snapshot 없을 때 Supabase `price_data` scan으로 내려간다 | `backend/app/repositories/market_repo.py:280`, `backend/app/repositories/market_repo.py:328`, `backend/app/repositories/market_repo.py:353` | `limit=500`이면 최대 25,000 row scan까지 갈 수 있다. 대량 read는 아니지만 검색 트래픽이 반복되면 thin DB 정책과 충돌한다. |
| CP129-P2-6 | P2 | CSS와 class naming에 legacy/composite 잔재가 남아 있다 | `frontend/src/app/globals.css:17`, `frontend/src/app/globals.css:315`, `frontend/src/app/globals.css:1758`, `frontend/src/components/StockView.tsx:1166` | 사용자 데이터 오염은 아니지만 제품 구조를 읽는 개발자가 composite가 아직 주류 경로인지 오해할 수 있다. |

## 3. 파일/함수/라인 근거

| 파일 | 함수/영역 | 라인 | 판정 |
|---|---|---:|---|
| `frontend/src/components/StockView.tsx` | 제품 run 상수 | 84-85 | `PRODUCT_LINE_RUN_ID`, `PRODUCT_BAND_RUN_ID`가 프론트에 직접 박혀 있다. |
| `frontend/src/components/StockView.tsx` | `loadStockData` prediction history fetch | 778-794 | 최신 prediction과 rolling history 모두 hardcoded run_id로 조회한다. history는 제품 history endpoint가 아니라 기존 history client를 사용한다. |
| `frontend/src/components/StockView.tsx` | 1W 상태 처리 | 752-775 | `isPredictionTimeframeEnabled` 통과 뒤에도 `nextTimeframe !== "1D"`에서 즉시 “주간 AI 예측은 준비 중”으로 반환한다. |
| `frontend/src/components/StockView.tsx` | active history filtering | 919-929 | old `PredictionResult[]` history를 차트 입력으로 사용한다. product history response shape와 연결되어 있지 않다. |
| `frontend/src/components/BacktestView.tsx` | 제품 run 상수와 subset scanner | 18-23 | line/band run_id와 12개 ticker scanner가 상수다. |
| `frontend/src/components/BacktestView.tsx` | 전략 신호 history fetch | 591-614 | 전략 카드가 기존 prediction history endpoint를 사용한다. |
| `frontend/src/components/BacktestView.tsx` | 단일 티커 백테스트 history fetch | 1058-1065 | 상세 백테스트도 기존 prediction history endpoint를 사용한다. |
| `frontend/src/components/BacktestView.tsx` | subset 안내 | 1258-1263 | 12개 subset이라는 제한은 UI에 표시되어 있어 P1에서는 제외했다. |
| `frontend/src/api/client.ts` | `isPredictionTimeframeEnabled` | 171-172 | client 타입은 1D와 1W를 모두 prediction-enabled로 본다. |
| `frontend/src/api/client.ts` | `fetchPredictionHistory` | 231-242 | `/api/v1/stocks/{ticker}/predictions/history`만 제공한다. product-history client 함수는 찾지 못했다. |
| `frontend/src/components/Chart.tsx` | latest-only filter | 90-92 | latest-only row는 rolling history에서 제외한다. 이 guard 자체는 좋지만, frontend가 old history를 쓰는 구조와 충돌한다. |
| `frontend/src/components/Chart.tsx` | rolling history builder | 258-283 | product latest-only row를 제외하고 old prediction rows를 rolling history로 그린다. |
| `frontend/src/components/Chart.tsx` | representative horizon | 49, 223 | display horizon을 계약에서 읽지 않고 index 4로 고정한다. |
| `backend/app/routers/v1/stocks.py` | product history endpoint | 112-136 | 제품용 `/predictions/product-history`가 이미 있다. |
| `backend/app/routers/v1/stocks.py` | legacy history endpoint | 140-153 | 기존 `/predictions/history`도 남아 있으며 frontend가 이쪽을 사용한다. |
| `backend/app/services/product_prediction_history_svc.py` | `get_product_prediction_history_data` | 168-221 | local parquet 기반 line/band product history response를 반환한다. 현재 frontend shape와 연결되지 않았다. |
| `backend/app/repositories/prediction_repo.py` | `fetch_prediction_by_run` | 44-63 | ticker와 run_id만 필터링한다. timeframe/horizon/role/meta guard가 없다. |
| `backend/app/repositories/prediction_repo.py` | `fetch_prediction_history_by_run` | 70-86 | history도 ticker와 run_id만 필터링한다. |
| `backend/collector/jobs/compute_market_breadth.py` | repair mode price read | 78-83 | `price_data`에서 `date,ticker,close`만 읽고 source/provider filter가 없다. |
| `backend/collector/jobs/compute_market_breadth.py` | incremental price read | 163-168 | 동일하게 source/provider filter가 없다. |
| `backend/collector/jobs/sync_sector_returns.py` | price read | 22-27 | `price_data`에서 source/provider filter 없이 sector return을 계산한다. |
| `backend/collector/pipelines/bootstrap_backfill.py` | breadth step | 281-287 | bootstrap backfill에서 provider 인자 없이 `run_market_breadth`를 호출한다. |
| `backend/collector/repositories/base.py` | bulk read guard | 42-56 | local snapshot required일 때 `price_data`, `indicators` 무제한 read를 막는 guard는 있다. 좋은 구조다. |
| `backend/app/repositories/market_repo.py` | provider filter | 49-71, 117-136 | price/indicator 제품 API는 source-aware 필터가 들어가 있다. 좋은 구조다. |
| `backend/app/repositories/market_repo.py` | search fallback | 280-363 | local snapshot이 없으면 stock_info scan 후 price_data fallback scan으로 내려간다. |
| `ai/storage.py` | latest-only guard | 53-109 | product latest-only 저장 함수는 row 수, 단일 asof, composite, layer를 검사한다. 좋은 구조다. |
| `ai/inference.py` | CLI save option | 66-70 | split과 save를 사용자가 직접 조합할 수 있다. |
| `ai/inference.py` | 일반 save path | 478-480 | product guard가 아니라 `save_predictions`, `save_prediction_evaluations`를 직접 호출한다. |
| `ai/composite_inference.py` | legacy composite guard | 337-341, 763-765 | composite 저장은 기본 차단되어 있고 명시 플래그가 필요하다. 의도된 legacy로 판단한다. |
| `ai/train.py` | checkpoint save path | 1767-1769 | checkpoint가 `ai/artifacts/checkpoints` 로컬 경로에 저장된다. |
| `ai/train.py` | model_runs checkpoint_path | 2274, 2342 | model_runs에 로컬 checkpoint path 문자열이 저장된다. |
| `backend/collector/pipelines/bootstrap_snapshot.py` | legacy indicator source | 45-47 | snapshot bootstrap indicators source/provider가 EODHD로 고정되어 있다. |
| `.gitignore` | artifact ignore | 1-39 | `.env`, `__pycache__`, `.next`, local parquet, yfinance cookie DB, wandb, artifacts ignore가 있어 저장소 노출 방어는 대체로 좋다. |

## 4. 바로 고칠 수 있는 quick fix

| 우선순위 | quick fix | 막는 위험 |
|---:|---|---|
| P0 | `StockView`와 `BacktestView`에서 제품 rolling history는 `/predictions/product-history`만 사용하게 바꾼다. 기존 `/predictions/history`는 실험 상세나 legacy debug 화면에서만 허용한다. | test split/bulk history가 제품 history처럼 보이는 위험 |
| P0 | `ai/inference.py --save`는 기본적으로 product run_id에 대해 차단하고, 제품 저장은 `save_product_latest_predictions()`를 쓰는 별도 `--product-latest-only` 경로만 허용한다. | Supabase full split 재저장과 latest-only 계약 우회 |
| P0 | `compute_market_breadth`와 `sync_sector_returns`는 source/provider 인자를 받기 전까지 yfinance/EODHD 병렬 DB에서 실행하지 않도록 pipeline gate를 둔다. | provider 혼합 지표 생성 |
| P1 | 제품 run_id를 frontend 상수가 아니라 backend product registry 또는 config endpoint에서 읽게 한다. | 모델 교체와 rollback 때 UI/운영 스크립트 불일치 |
| P1 | prediction 조회에서 run_id뿐 아니라 `timeframe`, `horizon`, `model_name`, `meta.layer`, `meta.product_latest_only`를 함께 검증한다. | legacy/test/다른 horizon row 혼입 |
| P1 | 1W 상태 계약을 하나로 정한다. Phase 1에서 1W 미출시면 `isPredictionTimeframeEnabled()`도 1D만 true로 맞춘다. | 사용자 상태 혼란 |
| P2 | `SIGNAL_SCAN_TICKERS`는 product scanner API 전용 임시 상수임을 코드 주석과 UI에 유지하고, 500티커처럼 보이는 문구는 금지한다. | subset이 제품 universe처럼 굳는 위험 |

## 5. 다음 CP로 빼야 할 구조 개선

| 영역 | 구조 개선 |
|---|---|
| Product model registry | `line-1d`, `band-1d`, `line-1w`, `band-1w` 슬롯을 DB 또는 versioned JSON 계약으로 관리한다. run_id, role, timeframe, horizon, source_data_hash, artifact hash, status를 한곳에서 내려준다. |
| Prediction storage | `product_latest`, `product_rolling_history`, `experiment_history`를 API/저장소 레벨에서 분리한다. 적어도 product UI는 old `predictions/history`를 호출하지 못하게 한다. |
| Source-aware breadth/sector | `market_breadth`, `sector_returns`에 source/provider 컬럼을 추가하거나 provider별 local parquet로 분리한다. 현재처럼 `date` unique만 두면 yfinance와 EODHD 결과를 동시에 보존할 수 없다. |
| Artifact registry | checkpoint path만 저장하지 말고 artifact id, relative path, sha256, feature contract, ticker registry hash를 함께 저장한다. 배포 서버가 artifact를 찾지 못하면 product run을 `사용 중`으로 표시하지 않는다. |
| Legacy script 격리 | `bootstrap_snapshot.py`, CP99/CP101/CP116/CP128 계열 스크립트는 `scripts/archive` 또는 `scripts/product_ops`로 나누고, write 가능 스크립트에는 provider/source guard banner를 둔다. |
| Static guard | CI 또는 preflight에서 product component의 `fetchPredictionHistory`, product script의 직접 `save_predictions`, source 없는 `price_data` read, hardcoded product run_id를 grep으로 막는다. |
| Search serving | stock search는 `stock_info` 또는 local snapshot을 원칙으로 하고, Supabase `price_data` fallback은 명시 제한과 telemetry를 둔다. |

## 6. 고쳐도 안 되는 것 / 의도된 legacy

| 항목 | 유지 이유 | 주의 |
|---|---|---|
| `ai/composite_inference.py` | legacy composite 진단 도구로 명시되어 있고 저장도 기본 차단되어 있다. | 제품 경로에서 import하거나 자동 실행하지 않는다. |
| EODHD fallback 문서와 일부 legacy 경로 | yfinance 전체 universe 전환과 rollback 검증 전까지는 golden reference 역할이 남아 있다. | source/provider 없는 계산 경로와는 분리해야 한다. |
| 12개 strategy signal subset | 현재 UI가 subset임을 밝히므로 데모용 임시 구조로는 허용 가능하다. | “전체 500티커 스캐너”처럼 보이게 만들면 안 된다. |
| 1W/1M 준비 중/가격 전용 상태 | 제품 모델 검증이 끝나기 전에는 무리해서 예측을 보여주지 않는 편이 안전하다. | client 타입, 제품 슬롯, 문구는 같은 계약을 봐야 한다. |
| Supabase thin DB 정책 | full history를 Supabase에 되돌리는 것은 비용/egress/제품 신뢰 리스크가 크다. | rolling history는 local parquet product history를 API로 serving하는 방향이 맞다. |
| CP128 product rolling history parquet | 기존 test split row를 쓰지 않고 제품 checkpoint/local parquet로 만든 별도 archive라 방향은 좋다. | frontend 연결 전까지 old history endpoint와 섞으면 안 된다. |

## 7. Phase 1 마무리 전 필수 수정 체크리스트

| 상태 | 체크 항목 |
|---|---|
| 미완료 | 제품 주식 차트와 백테스트 전략 신호가 `/predictions/product-history` 또는 동등한 product history API만 사용한다. |
| 미완료 | 기존 `/predictions/history`는 제품 화면에서 제거하거나 legacy/debug 전용으로 명확히 격리한다. |
| 미완료 | `ai/inference.py --save`가 product run에 대해 full split 저장을 못 하게 막는다. |
| 미완료 | `market_breadth`와 `sector_returns`가 source/provider-aware로 바뀌거나, 병렬 provider DB에서는 실행되지 않도록 차단한다. |
| 미완료 | 제품 run_id를 frontend/script hardcode에서 product registry 계약으로 이동한다. |
| 미완료 | prediction 조회가 run_id 외에 timeframe/horizon/model/role/latest-only meta를 검증한다. |
| 미완료 | 1W의 “지원 가능”, “준비 중”, “제품 미출시” 상태가 client, 화면, TrainingView에서 하나로 정리된다. |
| 미완료 | checkpoint artifact가 로컬 path만으로 운영되지 않도록 artifact hash/registry/복구 절차가 생긴다. |
| 미완료 | legacy EODHD snapshot bootstrap과 CP write scripts가 실수로 product path에서 실행되지 않게 격리된다. |
| 확인됨 | `.gitignore`는 `.env`, `__pycache__`, `.next`, local parquet, yfinance cookie DB, wandb, artifacts를 대체로 막고 있다. |
| 확인됨 | composite inference 저장은 기본 차단되어 있고 명시 플래그가 필요하다. |
| 확인됨 | 이번 감사에서는 코드 수정, DB read/write, 모델 학습/inference, npm build, 서버 재시작을 실행하지 않았다. |
