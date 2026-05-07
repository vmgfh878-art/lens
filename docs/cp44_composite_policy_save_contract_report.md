CP44-M은 risk_first_lower_preserve 정책을 composite inference 저장 경로에 연결하고, 실제 DB 저장 계약을 5티커 save-run으로 닫은 CP다.

## 1. 목표

CP43-M에서 채택 후보가 된 `risk_first_lower_preserve`를 `ai.composite_inference`의 실제 저장 경로에 연결했다. 이번 CP는 성능 개선이 아니라 `model_runs`, `predictions`, `prediction_evaluations`, `backtest_results` 저장 계약 검증이다.

## 2. 코드 변경

| 파일 | 변경 |
|---|---|
| `ai/composite_inference.py` | `--composition-policy` 옵션 추가 |
| `ai/composite_inference.py` | `raw_composite`, `include_line_clamp`, `risk_first_lower_preserve` 지원 |
| `ai/composite_inference.py` | 기본값 `risk_first_lower_preserve` 적용 |
| `ai/composite_inference.py` | `predictions.meta.composition_policy` 저장 |
| `ai/composite_inference.py` | composite `model_runs.config.composition_policy` 저장 |

`line_centered_asymmetric`은 CP43-M에서 coverage 0.374468, upper breach 0.562766으로 탈락했으므로 저장 경로 지원 정책에서 제외했다.

## 3. Risk-first 정책 정의

raw return 공간에서 적용한다.

```text
lower = min(calibrated_lower, line)
upper = max(calibrated_upper, line)
```

하방 보수성 원칙상 lower band를 덜 위험하게 끌어올리지 않는다. line이 기존 lower보다 더 낮으면 lower를 line까지 확장한다. 이 정책은 `lower <= line <= upper`를 보장한다.

## 4. 실행 조건

| 항목 | 값 |
|---|---|
| line run_id | `patchtst-1D-41d584bcb3cb` |
| band run_id | `cnn_lstm-1D-76f363b84218` |
| composition run_id | `composite-1D-3a44b5e51ed2` |
| tickers | `A`, `AAPL`, `ABBV`, `ABNB`, `ABT` |
| split | `test` |
| max_rows | 5 |
| device | `cuda` |
| amp_dtype | `off` |
| lower_scale | 1.908173 |
| upper_scale | 1.378499 |
| composition_policy | `risk_first_lower_preserve` |

실행 명령:

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
  --composition-policy risk_first_lower_preserve `
  --output-json docs\cp44_composite_policy_save_contract_metrics.json `
  --save
```

## 5. 저장 결과

| 항목 | 결과 |
|---|---:|
| exit code | 0 |
| saved_to_db | true |
| model_runs rows | 1 |
| model_runs.status | `completed` |
| predictions rows | 5 |
| prediction_evaluations rows | 5 |
| backtest_results rows | 1 |
| predictions.meta.composition_policy | `risk_first_lower_preserve` |

## 6. 계약 검증

| 검증 | 결과 |
|---|---|
| lower <= upper | PASS |
| lower <= line <= upper | PASS |
| line_inside_band_ratio | 1.000000 |
| forecast_dates length | 5 |
| line_series length | 5 |
| lower_band_series length | 5 |
| upper_band_series length | 5 |
| conservative_series length | 5 |
| 필수 meta 필드 | PASS |

필수 meta 필드:

- `composition_policy`
- `line_model_run_id`
- `band_model_run_id`
- `line_model_name`
- `band_model_name`
- `band_calibration_method`
- `band_calibration_params`
- `prediction_composition_version`

## 7. 평가 지표

5티커 smoke 기준이므로 성능 판단용 수치가 아니라 저장 계약 확인용이다.

| 지표 | 값 |
|---|---:|
| coverage | 0.920000 |
| lower_breach_rate | 0.000000 |
| upper_breach_rate | 0.080000 |
| avg_band_width | 0.451184 |
| line_inside_band_ratio | 1.000000 |
| spearman_ic | 0.900000 |
| long_short_spread | 0.028346 |
| fee_adjusted_return | 0.026346 |

## 8. Backtest 저장

실행:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.backtest --run-id composite-1D-3a44b5e51ed2 --timeframe 1D --save
```

결과:

| 항목 | 값 |
|---|---:|
| strategy_name | `band_breakout_v1` |
| return_pct | 0.0 |
| fee_adjusted_return_pct | 0.0 |
| num_trades | 0 |
| backtest_results rows | 1 |

Composite smoke는 `HOLD` signal을 저장하므로 거래 0건은 정상이다.

## 9. 검증

| 검증 | 결과 |
|---|---|
| `backend.db.scripts.ensure_runtime_schema` | PASS |
| `py_compile ai\composite_inference.py ai\composite_policy_eval.py` | PASS |
| `unittest ai.tests.test_storage_contracts` | 3건 PASS |
| `unittest ai.tests.test_inference_backtest ai.tests.test_evaluation_targets` | 10건 PASS |

병렬로 묶어 실행한 테스트 프로세스 하나에서 Windows/PyTorch `c10.dll` 초기화 오류가 한 번 발생했지만, `torch` 단독 import와 분리 실행 테스트는 통과했다. 코드 회귀로 보지는 않는다.

## 10. 결론

CP44-M은 통과했다. Composite 저장 계약은 `risk_first_lower_preserve` 기준으로 닫힌 것으로 본다.

다음 CP는 full run이 아니라 CNN-LSTM band sweep으로 가는 것이 맞다. 현재 composite 저장 경로는 line/band run_id, calibration params, composition policy를 추적 가능하게 저장한다.

## 11. 산출물

- `ai/composite_inference.py`
- `docs/cp44_composite_policy_save_contract_metrics.json`
- `docs/cp44_composite_policy_save_contract_report.md`
