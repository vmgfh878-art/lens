# CP14-P AI Run 조회 API 보고서

## 1. 추가 API 목록

- `GET /api/v1/ai/runs`
  - 저장된 AI run 목록을 조회한다.
  - 기본 필터는 `model_name=patchtst`, `status=completed`, `limit=20`, `offset=0`이다.
- `GET /api/v1/ai/runs/{run_id}`
  - 단일 AI run 상세를 조회한다.
  - `include_config=true`일 때만 원본 `config`를 포함한다.
- `GET /api/v1/ai/runs/{run_id}/evaluations`
  - 특정 run의 평가 결과를 조회한다.
- `GET /api/v1/ai/runs/{run_id}/backtests`
  - 특정 run의 저장된 백테스트 결과를 조회한다.
- `GET /api/v1/stocks/{ticker}/predictions/latest?run_id=...`
  - `run_id`가 없으면 기존 최신 예측 조회를 유지한다.
  - `run_id`가 있으면 해당 run의 해당 ticker 예측을 조회한다.

## 2. response schema 요약

- `RunSummary`
  - `run_id`, `status`, `model_name`, `timeframe`, `horizon`, `created_at`
  - `model_ver`, `checkpoint_path`, `best_epoch`, `best_val_total`
  - `line_target_type`, `band_target_type`
- `RunDetail`
  - `RunSummary` 전체
  - `val_metrics`, `test_metrics`, `config_summary`, `wandb_run_id`
  - `include_config=true` 요청 시 `config`
- `EvaluationSummary`
  - `run_id`, `ticker`, `timeframe`, `asof_date`
  - `coverage`, `avg_band_width`, `direction_accuracy`, `mae`, `smape`
  - `spearman_ic`, `top_k_long_spread`, `top_k_short_spread`, `long_short_spread`
  - `fee_adjusted_return`, `fee_adjusted_sharpe`, `fee_adjusted_turnover`
- `BacktestSummary`
  - `run_id`, `strategy_name`, `timeframe`, `return_pct`, `sharpe`, `mdd`
  - `win_rate`, `profit_factor`, `num_trades`
  - `fee_adjusted_return_pct`, `fee_adjusted_sharpe`, `avg_turnover`
  - `meta`, `created_at`

없는 필드는 row 본문, `config`, `val_metrics`, `test_metrics`, `meta` 순서로 안전하게 찾고, 끝까지 없으면 `null`로 응답한다. JSON 응답 안정성을 위해 `NaN`과 무한대 값은 `null`로 변환한다.

## 3. 기존 prediction API 회귀 여부

기존 호출은 그대로 유지된다.

```http
GET /api/v1/stocks/AAPL/predictions/latest
```

이 호출은 기존처럼 `model=patchtst`, `timeframe=1D`, 기본 horizon 규칙을 사용해 최신 예측을 조회한다. 새 `run_id` 쿼리는 선택값이므로 기존 프론트 호출과 응답 구조를 바꾸지 않는다.

## 4. failed_nan 처리 방식

- `/api/v1/ai/runs`는 기본적으로 `completed`만 조회한다.
- `failed_nan` run은 `GET /api/v1/ai/runs?status=failed_nan`처럼 명시적으로 요청할 때만 목록에 포함된다.
- `predictions/latest?run_id=...`는 run 상태가 `completed`가 아니면 `409 RUN_NOT_COMPLETED`로 거부한다.
- 학습 실행, 백테스트 실행, DB schema 변경은 포함하지 않았다.

## 5. 테스트 결과

추가 및 수정 테스트 범위:

- AI run 목록 기본 조회가 `completed` 상태로 repository를 호출하는지 확인
- `status=failed_nan` 명시 조회 확인
- 없는 run 상세 404 확인
- run 상세의 `config_summary`와 `NaN` 안전 변환 확인
- evaluations 조회가 path의 `run_id`로 필터되는지 확인
- backtests 조회가 path의 `run_id`로 필터되는지 확인
- 기존 `predictions/latest` 회귀 테스트 유지
- `predictions/latest?run_id=...`가 특정 run 조회 경로를 쓰는지 확인
- `failed_nan` run_id 예측 조회 거부 확인

실행 결과:

```text
$env:PYTHONPATH='backend'; python -m unittest backend.tests.test_api backend.tests.test_services
Ran 25 tests in 0.062s
OK

$env:PYTHONPATH='backend'; python -m unittest discover backend\tests
Ran 37 tests in 0.164s
OK
```

## 6. 프론트 호출 예시

```ts
await fetch("/api/v1/ai/runs?model_name=patchtst&status=completed&limit=20");
await fetch("/api/v1/ai/runs/run-20260418-001");
await fetch("/api/v1/ai/runs/run-20260418-001?include_config=true");
await fetch("/api/v1/ai/runs/run-20260418-001/evaluations?ticker=AAPL&limit=100");
await fetch("/api/v1/ai/runs/run-20260418-001/backtests?strategy_name=band_breakout_v1");
await fetch("/api/v1/stocks/AAPL/predictions/latest?run_id=run-20260418-001");
```

## 7. 남은 TODO

- 프론트 화면에서 run 선택 상태와 평가/백테스트 테이블을 연결한다.
- 운영 DB에 실제로 저장된 `meta` 확장 필드가 늘어나면 repository select 컬럼을 확장한다.
- run 목록 total count가 필요해지면 Supabase count 옵션을 별도 추가한다.
