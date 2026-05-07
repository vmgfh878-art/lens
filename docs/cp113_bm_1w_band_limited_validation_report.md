# CP113-BM 1W AI 밴드 3epoch 제한 검증 보고서

## 1. 결론

- 상태: PASS
- 판정: watch
- 실행 run_id: `cnn_lstm-1W-fceff544d34f`
- 목적: CP112-BM 1W smoke 통과 이후 같은 yfinance local 1W snapshot 기준 3epoch 제한 검증
- 제품 저장 후보 여부: 아직 아님. save-run/product 저장 없이 watch 후보로만 유지한다.
- 핵심 이유: test 기준 coverage와 interval은 CP112보다 개선됐지만, checkpoint 선택은 epoch 1에 머물렀고 selected val coverage_abs_error가 초안 기준 0.07을 소폭 넘었다.

## 2. 실행 환경 및 금지사항

| 항목 | 값 |
|---|---|
| python | `C:\Users\user\lens\.venv\Scripts\python.exe` |
| torch | `2.11.0+cu128` |
| CUDA | 사용 |
| GPU | `NVIDIA GeForce RTX 5060 Ti` |
| W&B | off, `--no-wandb`, `WANDB_MODE=disabled` |
| save-run | off |
| DB write | 없음 |
| inference 저장 | 없음 |
| composite | 사용 안 함 |
| local snapshot | on |

학습 명령은 `.venv\Scripts\python.exe -m ai.train` 전경 실행으로 수행했다. 후처리에서 `train_process.json` 기록 단계에 `Resolve-Path` 실수가 있었지만, 학습은 exit code 0과 cleanup marker까지 정상 종료했다. process log는 stdout/local summary 기준으로 복구 기록했다.

## 3. 데이터 및 피처 게이트

| 항목 | 결과 |
|---|---|
| timeframe | `1W` |
| provider/source | `yfinance` / `yfinance` |
| feature_version | `v3_adjusted_ohlc` |
| source_data_hash | `0f185b48` |
| price snapshot | `C:\Users\user\lens\data\parquet\price_data_yfinance_1W.parquet` |
| indicator snapshot | `C:\Users\user\lens\data\parquet\indicators_yfinance_1W.parquet` |
| eligible tickers | 97 |
| train / val / test rows | 26,675 / 5,723 / 5,820 |
| feature non-finite | 0 |
| target non-finite | 0 |
| feature_set | `price_volatility_volume` |
| 실제 사용 feature 수 | 11 |
| MODEL_N_FEATURES | 36 유지 |
| `atr_ratio` 모델 feature 포함 | false |

## 4. 실행 결과

| 항목 | 값 |
|---|---:|
| epochs | 3 |
| batch_size | 128 |
| total_elapsed_seconds | 112.1929 |
| epoch_seconds | 22.3326 / 20.4740 / 20.6192 |
| vram_peak_allocated_mb | 282.03 |
| checkpoint_selection | `band_gate` |
| selected_epoch | 1 |
| selected_reason | `band_gate_eligible` |
| band_gate_pass | true |
| role | `band_model` |

Epoch별 gate trace:

| epoch | val coverage_abs_error | val interval_score | band_gate |
|---:|---:|---:|---|
| 1 | 0.075162 | 0.279230 | PASS |
| 2 | 0.084291 | 0.235276 | FAIL |
| 3 | 0.087961 | 0.224857 | FAIL |

loss와 interval score는 epoch 2~3에서 좋아졌지만, coverage_abs_error가 band_gate 허용 구간 밖으로 밀려 selected checkpoint는 epoch 1이다.

## 5. band_metrics

| metric | val selected | test |
|---|---:|---:|
| nominal_coverage | 0.700000 | 0.700000 |
| empirical_coverage | 0.775162 | 0.715765 |
| coverage_abs_error | 0.075162 | 0.015765 |
| lower_breach_rate | 0.101913 | 0.135653 |
| upper_breach_rate | 0.122925 | 0.148582 |
| avg_band_width | 0.202463 | 0.196124 |
| median_band_width | 0.193701 | 0.184495 |
| p90_band_width | 0.334805 | 0.333164 |
| asymmetric_interval_score | 0.279230 | 0.317112 |
| interval_lower_penalty | 0.044977 | 0.077152 |
| interval_upper_penalty | 0.031791 | 0.043836 |
| band_width_ic | 0.211201 | 0.208742 |
| downside_width_ic | 0.019227 | 0.020930 |
| width_bucket_realized_vol_ratio | 1.672445 | 1.636228 |
| width_bucket_downside_rate_ratio | 0.956549 | 0.974899 |
| squeeze_breakout_rate | 0.075999 | 0.107388 |

`horizon=4`라 evaluator의 `h1_h5_band_*` bucket이 사실상 h1~h4 관찰치다. 해당 bucket 값은 all-horizon test band_metrics와 동일하게 기록됐다.

## 6. CP112-BM 대비

| metric | CP112 test | CP113 test | 변화 |
|---|---:|---:|---:|
| empirical_coverage | 0.731572 | 0.715765 | -0.015808 |
| coverage_abs_error | 0.031572 | 0.015765 | -0.015808 |
| lower_breach_rate | 0.130069 | 0.135653 | +0.005584 |
| upper_breach_rate | 0.138359 | 0.148582 | +0.010223 |
| avg_band_width | 0.246945 | 0.196124 | -0.050821 |
| median_band_width | 0.233818 | 0.184495 | -0.049323 |
| p90_band_width | 0.420838 | 0.333164 | -0.087674 |
| asymmetric_interval_score | 0.373170 | 0.317112 | -0.056058 |
| band_width_ic | 0.172422 | 0.208742 | +0.036320 |
| downside_width_ic | 0.017060 | 0.020930 | +0.003870 |
| width_bucket_realized_vol_ratio | 1.518010 | 1.636228 | +0.118217 |
| squeeze_breakout_rate | 0.129725 | 0.107388 | -0.022337 |

개선:

- test coverage_abs_error가 낮아졌다.
- 평균/중앙/p90 폭이 모두 줄었는데 interval score도 좋아졌다.
- band_width_ic와 downside_width_ic가 모두 양수이고 CP112보다 개선됐다.
- squeeze_breakout_rate가 낮아졌다.

주의:

- val selected coverage_abs_error는 0.075162로 초안 기준 0.07을 소폭 넘는다.
- epoch 2~3은 interval score는 더 좋아졌지만 coverage_abs_error 때문에 band_gate fail이다.
- downside_width_ic는 양수지만 절대값이 작아 하방 위험 적응력은 아직 약하다.

## 7. 1D 밴드와 다르게 보이는 점

참고 기준은 `docs/cp72_bm_1d_full_band_product_candidate_metrics.json`의 1D H5 5epoch full candidate다. CP113은 1W H4 3epoch 제한 검증이므로 직접 동등 비교는 아니다.

| metric | 1D CP72 | 1W CP113 test | 관찰 |
|---|---:|---:|---|
| coverage_abs_error | 0.014127 | 0.015765 | calibration은 비슷한 수준 |
| avg_band_width | 0.060732 | 0.196124 | 1W 절대 폭이 훨씬 큼 |
| asymmetric_interval_score | 0.124447 | 0.317112 | 1W interval cost가 큼 |
| band_width_ic | 0.372417 | 0.208742 | 1W dynamic width 신호가 약함 |
| downside_width_ic | 0.067348 | 0.020930 | 1W 하방 폭 적응력이 약함 |
| width_bucket_realized_vol_ratio | 2.491292 | 1.636228 | 1W도 양호하지만 1D보다 약함 |
| squeeze_breakout_rate | 0.045400 | 0.107388 | 1W squeeze 이후 breakout이 더 높음 |

요약하면 1W는 주간 수익률 스케일 때문에 band 폭과 interval score가 커진다. coverage calibration은 꽤 안정적이지만, 1D 대비 폭의 동적 정렬과 downside width 신호는 약하다.

## 8. 후보성 판단

판정: 1W band candidate watch

근거:

- test coverage_abs_error 0.015765로 초안 기준 0.07 이내다.
- lower/upper breach imbalance는 약 0.01293으로 과도하지 않다.
- band_width_ic 0.208742로 양수다.
- downside_width_ic 0.020930도 양수지만 낮다.
- CP112보다 폭, interval score, dynamic width proxy가 모두 개선됐다.

보류 이유:

- selected val coverage_abs_error가 0.07을 넘는다.
- 더 학습된 epoch 2~3이 gate를 통과하지 못해, 3epoch 학습 이득이 checkpoint 선택으로 온전히 연결되지 않았다.
- 1D 후보 대비 dynamic/downside width 신호가 약하다.

## 9. 다음 제안

- 바로 save-run 후보로 올리지 않는다.
- 같은 조건에서 seed 1~2개만 더 제한 검증해 CP113 개선이 seed 우연인지 확인한다.
- q 조정은 `q15/q85`를 유지한 재현성 확인 후 검토한다. q20/q80은 폭을 더 줄일 수 있지만 coverage와 lower breach가 흔들릴 수 있어 watch matrix로만 둔다.
- 1W 전용으로는 하방 폭 신호를 키울 수 있는 피처 또는 target/gate 쪽 검토가 필요하다. 단 이번 CP에서는 feature contract 변경과 `atr_ratio` 승격은 하지 않았다.

## 10. 산출물 및 검증

- `docs/cp113_bm_1w_band_limited_validation_report.md`
- `docs/cp113_bm_1w_band_limited_validation_metrics.json`
- `docs/cp113_bm_1w_band_limited_validation_logs/`

검증 수행 항목:

- metrics JSON 파싱
- preflight/process JSON 파싱
- 최종 python/pythonw 확인: `[]`, CP113 학습 프로세스 잔여 없음
- 최종 GPU compute 확인: visible python CUDA compute process 없음
