CP40-M은 200티커 확장이 아니라, PatchTST 예측선과 CNN-LSTM 보정 밴드를 조합하는 저장/추론 계약 CP다.

## 1. 변경 파일

- `ai/composite_inference.py`: PatchTST line checkpoint와 CNN-LSTM band checkpoint를 함께 로드해 조합 prediction row를 생성하는 smoke 도구를 추가했다.
- `ai/inference.py`: checkpoint 복원 시 PatchTST geometry와 CNN-LSTM `fp32_modules`가 누락되지 않도록 로더 인자를 보강했다.
- `backend/db/schema.sql`: `predictions.meta JSONB NOT NULL DEFAULT '{}'::jsonb`를 추가했다.
- `backend/db/scripts/ensure_runtime_schema.py`: 운영 DB에도 `predictions.meta`를 보장하도록 runtime schema ensure를 확장했다.
- `docs/cp40_line_band_composition_contract_metrics.json`: 5티커 조합 smoke 결과를 저장했다.

## 2. 저장 계약 결론

기존 `predictions`는 `run_id`, `model_name`, `model_ver`, band series는 저장할 수 있었지만, 조합 추론에 필요한 `line_model_run_id`, `band_model_run_id`, calibration params를 구조적으로 담을 위치가 없었다.

최소 변경안은 `predictions.meta JSONB`다. 개별 컬럼 추가는 쿼리 편의성은 좋지만 CP40 범위를 넘고, 기존 문자열 컬럼에 메타를 인코딩하는 방식은 검증성과 재현성이 떨어진다.

## 3. 조합 추론 계약

- `line_series`: PatchTST q25-b2 line checkpoint 출력.
- `lower_band_series`, `upper_band_series`: CNN-LSTM seq60 q20-b2 direct band checkpoint 출력에 scalar width calibration 적용.
- `conservative_series`: long-only 기준 보수 예측선으로 calibrated lower band를 사용.
- `signal`: 이번 CP에서는 성능/시그널 CP가 아니므로 `HOLD`로 고정했다.
- `forecast_dates`: line/band 모델의 `ticker`, `asof_date`, `forecast_dates`가 정확히 일치하는 행만 조합한다.

## 4. 사용 체크포인트

| 역할 | 체크포인트 | seq_len | target |
|---|---|---:|---|
| line_model | `ai/artifacts/checkpoints/patchtst_1D_patchtst-1D-19103d294e6b.pt` | 252 | `raw_future_return` |
| band_model | `ai/artifacts/checkpoints/cnn_lstm_1D_cnn_lstm-1D-b882658d1561.pt` | 60 | `raw_future_return` |

두 checkpoint는 `feature_version=v3_adjusted_ohlc`, `timeframe=1D`, `horizon=5`, `raw_future_return` 계약이 일치한다. `seq_len`은 달라도 출력 계약이 같으므로 조합 가능하다.

## 5. Calibration

CP39의 CNN-LSTM 100티커 scalar width calibration 계수를 그대로 사용했다.

| 항목 | 값 |
|---|---:|
| target_coverage | 0.85 |
| lower_scale | 1.908173 |
| upper_scale | 1.378499 |

calibration은 CNN-LSTM line 기준 lower/upper width에 적용하고, 조합 line은 PatchTST line을 사용한다. 따라서 `lower <= line <= upper`는 강제하지 않고 기록만 한다.

## 6. Smoke 실행

대상은 `A`, `AAPL`, `ABBV`, `ABNB`, `ABT` 5티커, split은 `test`, horizon은 5다.

실행은 venv에서 수행했다. 초기에는 `pandas`가 `torch`보다 먼저 로드되면서 Windows DLL 초기화 오류가 한 번 재현됐고, 이를 피하기 위해 `ai/composite_inference.py`의 import 순서를 `torch` 우선으로 정리했다. 수정 후 일반 `python -m ai.composite_inference` 경로로 exit code 0을 확인했다.

## 7. 계약 검증 결과

| 항목 | 결과 |
|---|---|
| row_count | 5 |
| line length = 5 | PASS |
| lower length = 5 | PASS |
| upper length = 5 | PASS |
| conservative length = 5 | PASS |
| lower <= upper | PASS |
| 필수 meta 포함 | PASS |
| lower <= line <= upper | 기록만 함, 일부 false |

`lower <= line <= upper`가 일부 false인 것은 CP40 지시상 실패 조건이 아니다. line과 band가 서로 다른 모델에서 오므로 이후 조합 calibration 또는 conservative policy에서 별도 판단해야 한다.

## 8. 조합 평가 지표

| 지표 | 값 |
|---|---:|
| coverage | 0.920000 |
| lower_breach_rate | 0.040000 |
| upper_breach_rate | 0.040000 |
| avg_band_width | 0.290036 |
| line MAE | 0.078195 |
| line SMAPE | 1.878300 |
| spearman_ic | 0.000000 |
| long_short_spread | -0.031182 |
| fee_adjusted_return | -0.033182 |

5티커 1개 as-of smoke라 투자 지표는 성능 판단용이 아니다. 이번 CP의 성공 기준은 조합 row와 저장 메타 계약 검증이다.

## 9. Prediction Meta

각 smoke row의 `meta`에는 다음 필드가 포함된다.

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

CP39 checkpoint는 `--save-run` 산출물이 아니라 checkpoint config의 `run_id`가 비어 있어, smoke에서는 `checkpoint:<checkpoint_stem>` 형식의 synthetic run ref를 사용했다. 실제 저장 run에서는 DB `model_runs.run_id`를 사용해야 한다.

## 10. DB 저장 상태

코드/스키마 레벨에서는 `predictions.meta` 저장 준비를 마쳤다. 다만 이번 CP에서는 `--save-run` 금지와 DB migration 미실행 조건 때문에 실제 운영 DB insert는 하지 않았다.

운영 DB 반영 전 필요한 명령:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m backend.db.scripts.ensure_runtime_schema
```

이후 `save_predictions()`는 `meta` 키가 포함된 prediction record를 그대로 upsert할 수 있다. unique key가 `run_id,ticker,model_name,timeframe,horizon,asof_date`라서 다른 run의 prediction row를 덮어쓰지 않는 정책도 유지된다.

## 11. 남은 리스크

- 실제 DB의 `predictions.meta` 존재 여부는 아직 확인하지 않았다.
- CP39 checkpoint는 synthetic run ref를 사용하므로, production 조합에는 저장된 line/band `model_runs.run_id`가 필요하다.
- line과 band가 다른 모델에서 오므로 `line_inside_band=false`가 발생할 수 있다. 이번 CP에서는 허용했지만, 다음 단계에서 line-centered band recentering 여부를 별도 결정해야 한다.
- Windows 환경에서는 `torch`를 `pandas`보다 먼저 import해야 했다. 현재 `ai/composite_inference.py`는 이 순서를 반영해 일반 모듈 실행이 가능하다.

## 12. 결론

CP40의 조합 저장/추론 계약은 작은 smoke 기준으로 통과했다. 다음 단계는 운영 DB에 `predictions.meta`를 적용한 뒤, 실제 `model_runs.run_id`를 가진 line/band 저장 run으로 조합 inference 저장 smoke를 한 번 더 닫는 것이다.
