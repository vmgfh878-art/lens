# CP112-BM 1W AI 밴드 스모크 보고서

## 1. 결론

- 상태: PASS
- 목적: yfinance local 1W snapshot 기준 CNN-LSTM AI 밴드 학습 파이프라인 smoke validation
- 성능 판정: 하지 않음. 이번 CP는 1W BM 후보의 데이터/shape/학습/metric 생성 확인이다.
- 실행 run_id: `cnn_lstm-1W-97f3b4d5b1ac`
- exit code: 0
- band_gate: PASS
- 실패 분류: 해당 없음. split/data/shape 실패 없이 학습과 평가가 종료됐다.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| python | `C:\Users\user\lens\.venv\Scripts\python.exe` |
| torch | `2.11.0+cu128` |
| CUDA | 사용 |
| GPU | `NVIDIA GeForce RTX 5060 Ti` |
| W&B | off, `--no-wandb`, `WANDB_MODE=disabled` |
| save-run | off |
| local snapshot | on |
| 추가 안전장치 | `LENS_REQUIRE_LOCAL_SNAPSHOTS=1` |

## 3. 데이터 사전 게이트

| 항목 | 결과 |
|---|---|
| timeframe | `1W` |
| provider/source | `yfinance` / `yfinance` |
| feature_version | `v3_adjusted_ohlc` |
| source_data_hash | `0f185b48` |
| price snapshot | `C:\Users\user\lens\data\parquet\price_data_yfinance_1W.parquet` |
| indicator snapshot | `C:\Users\user\lens\data\parquet\indicators_yfinance_1W.parquet` |
| usable rows | 53,300 |
| input ticker count | 100 |
| eligible ticker count | 97 |
| excluded ticker count | 3 |
| excluded tickers | `LMT`, `MS`, `T` |

## 4. 피처 계약 확인

| 항목 | 결과 |
|---|---|
| feature_set | `price_volatility_volume` |
| 실제 사용 feature 수 | 11 |
| MODEL_N_FEATURES | 36 유지 |
| MODEL_FEATURE_COLUMNS 수 | 36 |
| `atr_ratio` 모델 feature 포함 여부 | false |
| `atr_ratio` feature_set 포함 여부 | false |

사용 컬럼:

`log_return`, `open_ratio`, `high_ratio`, `low_ratio`, `vol_change`, `ma_5_ratio`, `ma_20_ratio`, `ma_60_ratio`, `rsi`, `macd_ratio`, `bb_position`

## 5. split 및 finite 확인

| split | rows | ticker | asof_min | asof_max | feature non-finite | target non-finite |
|---|---:|---:|---|---|---:|---:|
| train | 26,675 | 97 | 2018-02-09 | 2023-05-12 | 0 | 0 |
| val | 5,723 | 97 | 2023-08-11 | 2024-09-20 | 0 | 0 |
| test | 5,820 | 97 | 2024-12-20 | 2026-02-06 | 0 | 0 |

## 6. 학습 실행

| 항목 | 값 |
|---|---|
| model | `cnn_lstm` |
| horizon | 4 |
| seq_len | 104 |
| band_mode | `direct` |
| q_low / q_high | 0.15 / 0.85 |
| nominal_coverage | 0.70 |
| lambda_band | 2.0 |
| checkpoint_selection | `band_gate` |
| fp32_modules | `lstm,heads` |
| epochs | 1 |
| batch_size | 128 |
| amp_dtype | `bf16` |
| compile | false |
| elapsed_seconds | 66.7657 |
| epoch_seconds | 19.269 |
| estimated_remaining_seconds | 0.0 |
| vram_peak_allocated_mb | 282.03 |

## 7. band_metrics

검증 지표는 band_metrics만 기록했다. line metrics는 판정에 사용하지 않았다.

| metric | val | test |
|---|---:|---:|
| nominal_coverage | 0.700000 | 0.700000 |
| empirical_coverage | 0.775118 | 0.731572 |
| coverage_abs_error | 0.075118 | 0.031572 |
| lower_breach_rate | 0.099642 | 0.130069 |
| upper_breach_rate | 0.125240 | 0.138359 |
| avg_band_width | 0.252449 | 0.246945 |
| median_band_width | 0.242398 | 0.233818 |
| p90_band_width | 0.418098 | 0.420838 |
| asymmetric_interval_score | 0.342982 | 0.373170 |
| interval_lower_penalty | 0.049237 | 0.079246 |
| interval_upper_penalty | 0.041296 | 0.046979 |
| band_width_ic | 0.182265 | 0.172422 |
| downside_width_ic | 0.016405 | 0.017060 |
| width_bucket_realized_vol_ratio | 1.513183 | 1.518010 |
| width_bucket_downside_rate_ratio | 0.955084 | 0.970188 |
| squeeze_breakout_rate | 0.085171 | 0.129725 |

## 8. gate 결과

| 항목 | 값 |
|---|---|
| checkpoint_selection | `band_gate` |
| selected_epoch | 1 |
| selected_reason | `band_gate_eligible` |
| gate_failed | false |
| band_gate_pass | true |
| role | `band_model` |

## 9. 저장 및 금지사항 확인

- `--save-run`을 사용하지 않았다.
- W&B는 `--no-wandb`, `WANDB_MODE=disabled`로 비활성화했다. `wandb_status=disabled_by_cli`.
- DB write는 요청되지 않았다. `model_runs`, `predictions`, `prediction_evaluations` 저장 경로는 `save_run=false`로 실행되지 않았다.
- inference 저장은 수행하지 않았다.
- Supabase `price_data`/`indicators` 대량 read를 피하기 위해 local snapshot을 사용했고, `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`로 parquet 미사용 시 실패하도록 했다.
- 프론트/UI/backend 수정 없음.
- feature contract 변경 없음.
- `atr_ratio` 모델 feature 승격 없음.
- `ai.train` 기본 동작으로 로컬 checkpoint는 생성됐다: `C:\Users\user\lens\ai\artifacts\checkpoints\cnn_lstm_1W_cnn_lstm-1W-97f3b4d5b1ac.pt`

## 10. 검증

| 검증 | 결과 |
|---|---|
| `py_compile` preflight probe | PASS |
| `json.tool` metrics/preflight/process JSON | PASS |
| 최종 python/pythonw 확인 | CP112 신규 학습 프로세스 없음. 기존 PID 18244, 22544만 남음 |
| 최종 GPU 확인 | visible python CUDA compute process 없음 |

## 11. 산출물

- `docs/cp112_bm_1w_band_smoke_report.md`
- `docs/cp112_bm_1w_band_smoke_metrics.json`
- `docs/cp112_bm_1w_band_smoke_logs/`
