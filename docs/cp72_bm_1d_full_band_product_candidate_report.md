# CP72-BM 1D Band Full Product Candidate Report

## 결론

CP72-BM은 W&B online을 사용하지 않는 foreground 실행으로 closure 처리했다. 최종 제품 후보 run은 `cnn_lstm-1D-d0c780dee5e8`이며 `model_runs.status=completed`, `config.role=band_model`, `feature_set=price_volatility_volume`, `band_mode=direct`, `checkpoint_selection=band_gate`로 저장됐다.

제품 표시용 `predictions`와 `prediction_evaluations`는 test split 기준 각각 185,683건 저장됐다. checkpoint도 `ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-d0c780dee5e8.pt`에 존재한다.

## 실행/복구 기록

- W&B online run `run-20260501_115752-rdv0vhhw`는 `api.wandb.ai` 접속이 `127.0.0.1:9` proxy에서 connection refused로 막혀 실패/중단 run으로 기록했다.
- 해당 시도에서 생긴 partial DB row `cnn_lstm-1D-495d7eec9101`은 삭제하지 않았다. `status=failed_quality_gate`, predictions/evaluations 0건, checkpoint는 존재한다.
- W&B disabled 첫 재실행 `cnn_lstm-1D-5be94f33f784`는 학습은 완료됐지만 legacy band_gate가 q15/q85의 nominal coverage 0.70을 고정 0.85 목표로 판단해 `failed_quality_gate`가 됐다. 이 row도 삭제하지 않았다.
- `ai/train.py`의 band_gate를 `nominal_coverage=q_high-q_low` 기준으로 수정하고 재실행해 최종 completed run을 만들었다.

## 사전 게이트

| 항목 | 값 |
|---|---:|
| source data hash | `3ac43945` |
| feature_version | `v3_adjusted_ohlc` |
| feature_set | `price_volatility_volume` |
| feature_cache | `ai\cache\features_1D_4b3997385f0e_3ac43945.pt` |
| feature_index_cache | `ai\cache\feature_index_1D_42a1fd663092_3ac43945.pt` |
| input ticker | 503 |
| eligible ticker | 476 |
| no-limit actual training ticker | 476 |
| train / val / test samples | 863,202 / 184,789 / 185,683 |
| selected feature count | 11 |
| feature NaN/Inf | 0 |
| target NaN/Inf | 0 |
| atr_ratio 모델 feature | 미포함 |

## 최종 Run 설정

| 항목 | 값 |
|---|---|
| model | `cnn_lstm` |
| timeframe / horizon | `1D` / 5 |
| seq_len | 60 |
| q_low / q_high | 0.15 / 0.85 |
| lambda_band | 2.0 |
| batch_size / epochs | 256 / 5 |
| device / amp | `cuda` / `bf16` |
| fp32_modules | `lstm,heads` |
| W&B | disabled, `wandb_status=disabled_by_cli` |
| selected_epoch | 1 |
| selected_reason | `band_gate_eligible` |

## Band Metrics

| 지표 | 값 |
|---|---:|
| nominal_coverage | 0.7000 |
| empirical_coverage | 0.7141 |
| coverage_abs_error | 0.0141 |
| empirical_q_low | 0.1499 |
| empirical_q_high | 0.8641 |
| lower_breach_rate | 0.1499 |
| upper_breach_rate | 0.1359 |
| avg_band_width | 0.0607 |
| median_band_width | 0.0556 |
| p90_band_width | 0.0934 |
| asymmetric_interval_score | 0.1244 |
| interval_lower_penalty | 0.0446 |
| interval_upper_penalty | 0.0191 |
| band_width_ic | 0.3724 |
| downside_width_ic | 0.0673 |
| width_bucket_realized_vol_ratio | 2.4913 |
| width_bucket_downside_rate_ratio | 0.9610 |
| squeeze_breakout_rate | 0.0454 |

해석: coverage는 nominal 0.70 대비 +0.0141로 과도하지 않고, lower/upper breach 쏠림도 크지 않다. dynamic width는 `band_width_ic > 0`, `downside_width_ic > 0`, `width_bucket_realized_vol_ratio > 1` 조건을 만족한다.

## Baseline 비교

아래 baseline은 CP54의 1D h5 q15/q85 50티커 기준이라 CP72 full 476티커와 완전 동일 비교는 아니다. 방향성 참고로만 둔다.

| 기준 | coverage_abs_error | asymmetric_interval_score | band_width_ic | downside_width_ic | squeeze_breakout_rate |
|---|---:|---:|---:|---:|---:|
| CP72 final | 0.0141 | 0.1244 | 0.3724 | 0.0673 | 0.0454 |
| constant_width_train_quantile | 0.0409 | 0.1406 | 0.2682 | 0.0658 | 0.0835 |
| rolling_historical_quantile_band_w252 | 0.0263 | 0.1310 | 0.3780 | 0.0929 | 0.0550 |
| rolling_bollinger_return_band_w60_k1 | 0.0000 | 0.1336 | 0.3395 | 0.0641 | 0.0683 |

판정: CP72는 interval score와 squeeze_breakout_rate에서 세 baseline 참고값보다 좋다. coverage_abs_error는 constant/historical보다 좋지만 Bollinger w60 k1보다 나쁘다. dynamic width는 Bollinger보다 강하고, historical w252보다는 `band_width_ic`와 `downside_width_ic`가 약간 낮다.

## 저장 확인

| 항목 | 상태 |
|---|---|
| model_runs row | 존재 |
| predictions row | 185,683건 |
| prediction_evaluations row | 185,683건 |
| checkpoint | 존재 |
| role 식별 | `config.role=band_model` |
| feature_set 저장 | `price_volatility_volume` |
| q/lambda/band config 저장 | q_low=0.15, q_high=0.85, lambda_band=2.0, band_mode=direct |

## 코드 수정

- `ai/train.py`: band_gate가 fixed 0.85 coverage target을 쓰던 문제를 수정했다. 이제 `band_metrics.nominal_coverage` 기준 coverage error와 tail breach error로 판단한다.
- `ai/inference.py`: feature_set checkpoint의 `feature_columns/n_features`를 모델 복원과 입력 feature 선택에 반영했다. DataLoader 생성 전에 36개 feature를 11개로 줄이도록 순서를 고쳤다.

## 검증

- `.venv\Scripts\python.exe -m py_compile ai\train.py ai\inference.py ai\sweep.py ai\tests\test_checkpoint_selection.py ai\tests\test_wandb_optional_fallback.py ai\tests\test_inference_backtest.py`: 통과
- `.venv\Scripts\python.exe -m unittest ai.tests.test_checkpoint_selection ai.tests.test_wandb_optional_fallback ai.tests.test_inference_backtest`: 통과
- `.venv\Scripts\python.exe -m json.tool docs\cp72_bm_1d_full_band_product_candidate_metrics.json`: 통과
- 마지막 `python/pythonw` 잔여 프로세스: 없음

## 다음 CP

다음 순서는 CP64-D indicator full backfill 후속 정리 또는 CP72 제품 후보를 UI/backend에서 어떤 run으로 노출할지 연결 정책을 별도 CP로 진행하는 것이다. W&B online이 필요한 실험은 앞으로 에이전트가 실행하지 않고 명령만 제공한다.
