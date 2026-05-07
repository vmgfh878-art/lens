# CP31-S 저장 계약 스모크 보고서

## 1. 목표
CP29-D와 CP30-G 이후 실제 DB 저장 계약이 end-to-end로 동작하는지 작은 run으로 검증했다. 성능 판단이 아니라 `model_runs`, `predictions`, `prediction_evaluations`, `backtest_results` 저장과 checkpoint registry 재현성을 확인하는 작업이다.

## 2. 실행 요약
첫 저장 run은 `patchtst-1D-01ab0fc075f4`이고, 동일 key 보존 검증용 두 번째 run은 `patchtst-1D-239b58ab90f0`이다. 둘 다 PatchTST 5티커, 1D, 1epoch, `--save-run`, `--no-wandb`, `--no-compile`, raw return target 조건으로 실행했다.

## 3. 코드 수정
inference 저장 경로에서 두 문제가 발견되어 수정했다.

- `ai/inference.py`: Windows venv에 Triton이 없을 때 CUDA inference의 `torch.compile`을 건너뛰는 폴백을 추가했다.
- `ai/inference.py`: per-sample `PinballLoss` 계산에서 prediction은 CPU, target은 CUDA인 device mismatch를 수정했다. 평가용 `band_target`을 CPU tensor로 맞춘다.

## 4. model_runs 저장 확인
| 항목 | 확인값 |
|---|---|
| run_id | `patchtst-1D-01ab0fc075f4` |
| feature_version | `v3_adjusted_ohlc` |
| band_mode | `direct` |
| status | `completed` |
| checkpoint_path | `ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-01ab0fc075f4.pt` |
| ticker_registry_path | `ai\cache\ticker_id_map_1d_7fe18662241f.json` |
| registry_exists | true |
| num_tickers | 5 |

## 5. inference 저장 확인
`python -m ai.inference --run-id patchtst-1D-01ab0fc075f4 --split test --save` 재시도 결과 exit code 0으로 종료했다.

| 테이블 | 저장 row 수 | 확인 |
|---|---:|---|
| predictions | 1,635 | run_id 포함 저장 |
| prediction_evaluations | 1,635 | run_id 포함 저장 |

`prediction_evaluations`에는 `lower_breach_rate`, `upper_breach_rate`가 실제 값으로 저장됐다. 첫 샘플 `AAPL / 2024-10-03` 기준 두 값 모두 0.0으로 조회됐다.

## 6. backtest 저장 확인
`python -m ai.backtest --run-id patchtst-1D-01ab0fc075f4 --timeframe 1D --save` 실행 결과 exit code 0이었다.

| 항목 | 값 |
|---|---:|
| backtest_results rows | 1 |
| strategy_name | `band_breakout_v1` |
| return_pct | 51.074185 |
| mdd | -0.260514 |
| sharpe | 0.072150 |
| num_trades | 215 |
| portfolio_dates | 372 |

## 7. run_id별 prediction 보존 확인
같은 조건으로 두 번째 저장 run `patchtst-1D-239b58ab90f0`을 만들고 inference `--save`를 실행했다. 동일 `ticker=AAPL`, `asof_date=2024-10-03`에 대해 predictions와 prediction_evaluations 모두 run_id 2개가 별도 row로 남았다.

| 확인 항목 | 결과 |
|---|---|
| same key prediction rows | 2 |
| prediction run_ids | `patchtst-1D-01ab0fc075f4`, `patchtst-1D-239b58ab90f0` |
| same key evaluation rows | 2 |
| run별 prediction count | 1,635 / 1,635 |

따라서 `predictions`의 upsert key가 `run_id`를 포함하고 있으며, 다른 run이 같은 ticker/asof_date를 생성해도 기존 row를 덮어쓰지 않는 것이 확인됐다.

## 8. checkpoint registry 사용 확인
inference는 `model_runs.checkpoint_path`를 읽고 checkpoint config의 `ticker_registry_path`를 사용한다. 두 run 모두 checkpoint 내부 registry path가 `ai\cache\ticker_id_map_1d_7fe18662241f.json`으로 저장되어 있었고 파일도 존재했다. `num_tickers=5`, `feature_version=v3_adjusted_ohlc`도 checkpoint config와 일치했다.

## 9. 테스트
관련 회귀 테스트를 실행했다.

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m unittest ai.tests.test_inference_backtest ai.tests.test_storage_contracts ai.tests.test_ticker_registry
```

결과: 12 tests OK.

## 10. 결론
CP31-S 저장 계약 스모크는 통과했다. DB 저장 4종(`model_runs`, `predictions`, `prediction_evaluations`, `backtest_results`)이 end-to-end로 동작했고, checkpoint registry 기반 inference와 run_id별 prediction 보존도 확인됐다. 남은 이슈는 성능이 아니라 inference CUDA 환경에서 Triton이 없을 때 compile을 건너뛰는 폴백이 필요했다는 점이며, 해당 폴백은 코드에 반영했다.
