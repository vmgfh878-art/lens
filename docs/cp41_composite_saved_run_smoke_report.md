CP41-S는 CP40의 synthetic checkpoint smoke를 실제 model_runs.run_id 기반 저장 스모크로 닫는 CP다.

## 1. 목표

CP40-M에서는 `checkpoint:<stem>` synthetic ref로 line/band 조합 계약을 검증했다. CP41-S에서는 실제 `model_runs.run_id`가 있는 checkpoint를 사용해 composite `model_runs`, `predictions`, `prediction_evaluations`, `backtest_results` 저장까지 확인했다.

금지 조건은 지켰다. full 473티커, W&B sweep, UI 수정, 신규 모델 구현, schema 추가 변경은 하지 않았다. CP40에서 추가한 `predictions.meta` migration만 실행했다.

## 2. Migration

실행:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m backend.db.scripts.ensure_runtime_schema
```

결과:

| 항목 | 결과 |
|---|---|
| exit code | 0 |
| `predictions.meta` | 존재 확인 |
| prediction writer columns | `band_quantile_high`, `band_quantile_low`, `line_series`, `meta`, `run_id` |
| Supabase ready | `database=true` |

`backend.collector.pipelines.backfill_status`는 readiness 전체 집계 쿼리에서 Supabase `57014 statement timeout`이 발생했다. 이는 migration 오류가 아니라 기존 readiness 집계 쿼리 비용 문제로 보이며, 간단 DB ready check는 통과했다.

## 3. 실제 run_id 생성

처음 지시된 5티커/1epoch 조건은 line_gate와 band_gate가 모두 너무 짧아 `failed_quality_gate`가 발생했다. CP41의 핵심 조건은 `completed` 상태 run만 composite에 쓰는 것이므로, full run은 하지 않고 가장 작은 completed run을 확보하는 방향으로 조정했다.

| 역할 | 최초 시도 | 결과 | 최종 사용 run |
|---|---|---|---|
| line_model | 5티커 1epoch `line_gate` | `failed_quality_gate` | 50티커 1epoch |
| band_model | 5티커 1epoch `band_gate` | `failed_quality_gate` | 50티커 3epoch |

최종 사용 run:

| 역할 | run_id | status | checkpoint |
|---|---|---|---|
| line_model | `patchtst-1D-41d584bcb3cb` | `completed` | `ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-41d584bcb3cb.pt` |
| band_model | `cnn_lstm-1D-76f363b84218` | `completed` | `ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-76f363b84218.pt` |

## 4. Composite 저장 실행

실행:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.composite_inference `
  --line-run-id patchtst-1D-41d584bcb3cb `
  --band-run-id cnn_lstm-1D-76f363b84218 `
  --tickers A AAPL ABBV ABNB ABT `
  --max-rows 5 `
  --split test `
  --device cuda `
  --batch-size 256 `
  --amp-dtype off `
  --lower-scale 1.908172845840454 `
  --upper-scale 1.378499150276184 `
  --output-json docs\cp41_composite_saved_run_smoke_metrics.json `
  --save
```

결과:

| 항목 | 값 |
|---|---|
| composition_run_id | `composite-1D-a0786769a07a` |
| exit code | 0 |
| saved_to_db | true |
| row_count | 5 |

## 5. Prediction 저장 검증

| 항목 | 결과 |
|---|---|
| `model_runs.status` | `completed` |
| `predictions` rows | 5 |
| `prediction_evaluations` rows | 5 |
| `forecast_dates` length | 5 |
| `line_series` length | 5 |
| `lower_band_series` length | 5 |
| `upper_band_series` length | 5 |
| `conservative_series` length | 5 |
| `lower <= upper` | true |

`lower <= line <= upper`는 일부 false다. CP40/CP41 계약상 실패 조건이 아니라 기록 조건이다.

## 6. Meta 저장 검증

첫 prediction row의 `meta`에는 다음 키가 저장됐다.

- `line_model_run_id`
- `band_model_run_id`
- `line_model_name`
- `band_model_name`
- `line_checkpoint_path`
- `band_checkpoint_path`
- `band_calibration_method`
- `band_calibration_params`
- `prediction_composition_version`
- `line_seq_len`
- `band_seq_len`
- `target_contract`
- `validation`

필수 메타가 실제 `predictions.meta`에 들어간 것을 확인했다.

## 7. Composite 평가 지표

5티커 단일 smoke 기준이므로 성능 판단용 수치는 아니다.

| 지표 | 값 |
|---|---:|
| coverage | 0.800000 |
| lower_breach_rate | 0.040000 |
| upper_breach_rate | 0.160000 |
| avg_band_width | 0.354500 |
| mae | 0.275374 |
| smape | 1.860017 |
| spearman_ic | 0.900000 |
| long_short_spread | 0.028346 |
| fee_adjusted_return | 0.026346 |

## 8. Backtest 저장 검증

실행:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.backtest --run-id composite-1D-a0786769a07a --timeframe 1D --save
```

결과:

| 항목 | 값 |
|---|---:|
| backtest_results rows | 1 |
| strategy_name | `band_breakout_v1` |
| return_pct | 0.0 |
| num_trades | 0 |
| portfolio_dates | 1 |

이번 composite smoke는 signal을 `HOLD`로 저장하므로 trade가 없는 것은 정상이다. 목적은 backtest 저장 계약이 composite run_id를 받아들이는지 확인하는 것이다.

## 9. 코드 변경

- `ai/composite_inference.py`
  - `--line-run-id`, `--band-run-id`, `--composition-run-id`, `--save` 추가.
  - `model_runs.status='completed'` run만 입력으로 허용.
  - composite `model_runs` 저장 후 `predictions`, `prediction_evaluations` 저장.
  - `prediction_evaluations`는 composite output 기준 breach/coverage/MAE/SMAPE를 저장한다.

schema 추가 변경은 하지 않았다. CP40에서 준비한 `predictions.meta`를 migration으로 반영했을 뿐이다.

## 10. 검증

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m py_compile ai\composite_inference.py ai\inference.py ai\tests\test_storage_contracts.py backend\db\scripts\ensure_runtime_schema.py
```

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -c "import torch, unittest; ..."
```

결과: 관련 회귀 25건 통과.

## 11. 결론

CP41-S 저장 계약은 통과했다. synthetic checkpoint ref가 아니라 실제 `model_runs.run_id` 기반으로 line/band checkpoint를 조회했고, composite run `composite-1D-a0786769a07a`에 대해 `model_runs`, `predictions`, `prediction_evaluations`, `backtest_results` 저장이 모두 확인됐다.

다만 5티커/1epoch로는 line_gate/band_gate completed run이 나오지 않아, 실제 저장 smoke는 50티커 completed run으로 닫았다. 이건 성능 확장이 아니라 gate 정책과 저장 계약 사이의 현실적인 충돌을 피한 조정이다.
