CP48-M은 h5 후보 확장이 아니라, 원 기획 h_max=20 기준으로 1D horizon=20이 학습 가능한지 확인하는 feasibility smoke다.

## 범위

이번 CP는 50티커, 1epoch, `save-run` 없음, W&B 없음, UI 수정 없음으로 제한했다. h5 결과와 비교는 참고로만 사용했고, h20 결과를 h5 후보와 같은 후보군으로 섞지 않았다.

| 항목 | 값 |
|---|---|
| feature_version | `v3_adjusted_ohlc` |
| timeframe | `1D` |
| horizon | 20 |
| limit_tickers | 50 |
| eligible_tickers | 48 |
| target | `raw_future_return` |
| PatchTST selector | `line_gate` |
| CNN-LSTM selector | `band_gate` |

## PatchTST line h20 smoke

| 항목 | 값 |
|---|---:|
| run_id | `patchtst-1D-7905007e59e3` |
| exit code | 0 |
| epoch_seconds | 76.1584 |
| VRAM peak MB | 5155.34 |
| line_gate_pass | false |
| selected_reason | `line_gate_failed_fallback_val_total` |

| split | coverage | upper breach | lower breach | avg width | spearman_ic | long_short_spread |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.999664 | 0.000200 | 0.000136 | 0.971184 | -0.001940 | 0.005621 |
| test | 0.999738 | 0.000131 | 0.000131 | 1.040177 | -0.047750 | -0.013304 |

판단: h20 PatchTST line은 실행 가능하지만, 1epoch 기준으로 IC가 음수이고 `line_gate`가 실패했다. 밴드는 거의 모든 값을 덮는 상태라 h20 line 후보로 바로 올릴 수 없다.

## CNN-LSTM band h20 smoke

| 항목 | 값 |
|---|---:|
| run_id | `cnn_lstm-1D-19a36c1ee93e` |
| exit code | 0 |
| epoch_seconds | 70.3588 |
| VRAM peak MB | 1295.85 |
| band_gate_pass | false |
| selected_reason | `band_gate_failed_fallback_val_total` |

| split | coverage | upper breach | lower breach | avg width | spearman_ic | long_short_spread |
|---|---:|---:|---:|---:|---:|---:|
| raw validation | 0.668474 | 0.108754 | 0.222772 | 0.176947 | -0.087366 | -0.026499 |
| raw test | 0.546591 | 0.128553 | 0.324856 | 0.169311 | -0.000508 | 0.008080 |

raw band는 coverage가 낮고 lower breach가 너무 커서 실패다.

## Band scalar calibration

CNN-LSTM h20 band에 validation 기준 scalar width calibration을 적용했다.

| 항목 | 값 |
|---|---:|
| lower_scale | 2.466125 |
| upper_scale | 1.079088 |
| target coverage | 0.85 |

| split | coverage | upper breach | lower breach | avg width |
|---|---:|---:|---:|---:|
| validation | 0.850000 | 0.075000 | 0.075000 | 0.309498 |
| test | 0.801636 | 0.076827 | 0.121537 | 0.321402 |

판단: h20 band는 raw 상태에서는 실패지만 scalar calibration 후에는 test coverage 0.8016까지 회복된다. 다만 lower breach 0.1215는 기존 기준 0.12를 살짝 넘는다.

## Composite h20 probe

같은 h20 PatchTST line checkpoint와 CNN-LSTM band checkpoint를 `risk_first_lower_preserve`로 조합했다. DB 저장은 하지 않았다.

| 항목 | 값 |
|---|---:|
| row_count | 48 |
| forecast_dates length | 20 |
| line_series length | 20 |
| lower_band_series length | 20 |
| upper_band_series length | 20 |
| conservative_series length | 20 |
| lower <= upper | true |
| line_inside_band_ratio | 1.000000 |

| coverage | upper breach | lower breach | avg width | spearman_ic | long_short_spread |
|---:|---:|---:|---:|---:|---:|
| 0.728125 | 0.268750 | 0.003125 | 0.427440 | -0.165436 | -0.067076 |

Composite 계약은 horizon=20 길이로 동작한다. 다만 upper breach가 0.2688로 너무 높고 line 지표가 음수라 h20 composite 후보로는 아직 불합격이다.

참고: 기존 `composite_inference`의 contract check 이름이 `series_length_all_5`로 하드코딩되어 있어 h20에서는 해당 체크가 false로 찍힌다. 실제 series 길이는 모두 20으로 확인했다.

## h5 참고 비교

아래 값은 CP42/CP45의 188티커, 3epoch 기준이다. 이번 h20 50티커 1epoch와 같은 후보로 섞지 않는다.

| 항목 | h5 참고 | h20 smoke |
|---|---:|---:|
| PatchTST line epoch_seconds | 301~310 | 76.16 |
| PatchTST line VRAM MB | 5153 | 5155 |
| PatchTST test spearman_ic | 0.028802 | -0.047750 |
| PatchTST test long_short_spread | 0.001868 | -0.013304 |
| CNN-LSTM band epoch_seconds | 113.13 | 70.36 |
| CNN-LSTM band VRAM MB | 323.98 | 1295.85 |
| CNN-LSTM scalar test coverage | 0.808130 | 0.801636 |
| CNN-LSTM scalar test avg width | 0.098938 | 0.321402 |

h20 band 폭은 h5 대비 약 3.25배 넓다. horizon이 길어진 만큼 자연스러운 부분도 있지만, line과 composite 지표가 동시에 무너져 즉시 후보로 올리기는 어렵다.

## 판단

h20 학습 자체는 가능하다. CUDA, bf16, checkpoint 생성, calibration, composite probe 모두 exit code 0으로 끝났다.

하지만 h20 PatchTST line은 `line_gate` 실패, h20 CNN-LSTM raw band는 `band_gate` 실패, h20 composite는 upper breach와 IC/spread가 무너졌다.

따라서 현재 결정은 다음과 같다.

| 항목 | 판단 |
|---|---|
| h5 단기 모델 | 유지 |
| h20 full run | 금지 유지 |
| h20 branch | Phase 1.5로 분리 |
| h20 추가 sweep | W&B 켜고 별도 branch에서 3epoch 이상으로만 검토 |

## 다음 권장

h20을 당장 CP49 본류로 올리지 않는다. 다음 본류는 h5 composite 저장/검증을 계속 진행하고, h20은 별도 branch에서 다음 순서로만 재개하는 것이 맞다.

1. h20 PatchTST line 3epoch, 50티커 재확인
2. h20 CNN-LSTM band q10/q90 또는 conformal calibration 비교
3. h20 composite upper buffer 정책 재적용
4. 세 항목 중 하나라도 통과하면 W&B on으로 h20 전용 소규모 sweep
