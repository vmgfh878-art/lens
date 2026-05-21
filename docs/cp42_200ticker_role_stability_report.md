CP42-M은 실제 저장 계약이 닫힌 상태에서 line 후보와 band 후보를 200티커로 제한 검증한 CP다.

## 1. 목표

CP41-S에서 실제 `model_runs.run_id` 기반 composite 저장 계약을 닫았으므로, CP42-M에서는 full 473티커 직전 안정성 확인을 했다. 이번 CP는 저장 CP가 아니므로 `--save-run`을 쓰지 않았고, schema 변경도 하지 않았다.

금지 조건 준수:

- full 473티커 실행 안 함
- W&B sweep 안 함
- UI 수정 안 함
- schema 변경 안 함
- 신규 모델 구현 안 함

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| device | `cuda` |
| no_compile | true |
| no_wandb | true |
| save_run | false |
| feature_version | `v3_adjusted_ohlc` |
| timeframe | `1D` |
| requested limit_tickers | 200 |
| eligible_tickers | 188 |

`limit_tickers=200`을 요청했지만 8분기 재무 gate 등으로 188개가 실제 학습 대상이 됐다.

## 3. PatchTST line 200티커

설정:

| 항목 | 값 |
|---|---|
| model | `patchtst` |
| role | `line_model` |
| checkpoint_selection | `line_gate` |
| seq_len | 252 |
| patch_len / patch_stride | 16 / 8 |
| q_low / q_high | 0.25 / 0.75 |
| lambda_band | 2.0 |
| batch_size | 256 |
| epochs | 3 |

결과:

| split | selected_epoch | line_gate | spearman_ic | long_short_spread | mae | smape | direction_accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
| validation | 1 | PASS | 0.069564 | 0.012140 | 0.046873 | 1.505758 | 0.502540 |
| test | - | 참고 | 0.028802 | 0.001868 | 0.050826 | 1.493697 | 0.506413 |

판정:

- validation/test 모두 `spearman_ic > 0`.
- validation/test 모두 `long_short_spread > 0`.
- PatchTST line 후보는 200티커에서 생존이다.

속도:

| epoch | seconds |
|---:|---:|
| 1 | 310.4414 |
| 2 | 309.0915 |
| 3 | 301.3499 |

VRAM peak는 약 5153 MB였다.

## 4. CNN-LSTM band 200티커 원본

설정:

| 항목 | 값 |
|---|---|
| model | `cnn_lstm` |
| role | `band_model` |
| checkpoint_selection | `band_gate` |
| seq_len | 60 |
| q_low / q_high | 0.20 / 0.80 |
| lambda_band | 2.0 |
| band_mode | `direct` |
| fp32_modules | `lstm,heads` |
| batch_size | 256 |
| epochs | 3 |

원본 결과:

| split | band_gate | coverage | lower_breach | upper_breach | avg_band_width | band_loss |
|---|---|---:|---:|---:|---:|---:|
| validation | FAIL | 0.528572 | 0.268288 | 0.203140 | 0.036534 | 0.016109 |
| test | 참고 | 0.483245 | 0.285051 | 0.231704 | 0.040410 | 0.021954 |

원본 band는 너무 좁아서 탈락이다. 이 결과는 CP39와 같은 방향이며, CNN-LSTM band는 scalar width calibration 없이는 쓰면 안 된다.

속도:

| epoch | seconds |
|---:|---:|
| 1 | 103.0504 |
| 2 | 96.1813 |
| 3 | 98.8249 |

VRAM peak는 약 324 MB였다.

## 5. Scalar Width Calibration

validation split에서 target coverage 0.85 기준으로 scale을 fit하고 test split에 적용했다.

| 항목 | 값 |
|---|---:|
| lower_scale | 2.731744 |
| upper_scale | 1.767985 |
| target_coverage | 0.85 |

calibrated 결과:

| split | coverage | lower_breach | upper_breach | avg_band_width | band_loss |
|---|---:|---:|---:|---:|---:|
| validation | 0.849999 | 0.075001 | 0.075001 | 0.078502 | 0.018552 |
| test | 0.830066 | 0.084131 | 0.085802 | 0.119297 | 0.028004 |

판정:

- calibrated test coverage 0.830066: 통과.
- upper_breach 0.085802: 통과.
- lower_breach 0.084131: 통과.
- validation/test coverage gap은 약 0.0199로 허용 가능.
- avg_band_width는 원본 대비 넓어졌지만 breach 안정성을 위해 필요한 수준으로 보인다.

CNN-LSTM band 후보는 200티커에서 scalar calibration 전제 생존이다.

## 6. Composite 가능성 Probe

실제 DB 저장은 하지 않고, 200 limit 조건으로 line/band 조합 가능성만 측정했다.

| 항목 | 값 |
|---|---:|
| row_count | 188 |
| lower <= upper | true |
| line_inside_band_count | 97 |
| line_inside_band_ratio | 0.515957 |
| composite coverage | 0.836170 |
| composite upper_breach | 0.156383 |
| composite lower_breach | 0.007447 |
| composite avg_band_width | 0.286390 |

해석:

- `lower <= upper`는 모두 통과했다.
- `line_inside_band_ratio`는 약 51.6%다.
- line과 band가 서로 다른 모델이라 중심선이 밴드 밖에 있는 경우가 절반 가까이 발생한다.
- composite coverage는 0.836으로 좋지만 upper breach가 0.156으로 기준 0.15를 아주 약간 넘는다.

## 7. Composite 계약 적용 가능성

CP41 계약에는 들어갈 수 있다.

- line/band 모두 `v3_adjusted_ohlc`, `1D`, `horizon=5`, `raw_future_return` 계약을 공유한다.
- seq_len은 PatchTST 252, CNN-LSTM 60으로 다르지만 inference output 공간이 같으므로 조합 가능하다.
- conservative_series는 계속 calibrated lower band로 둔다.
- `line_inside_band=false`를 실패로 두지 않는 CP40/41 계약은 200티커에서도 유지해야 한다.

다만 다음 정책 결정이 필요하다.

- line을 band 중앙으로 recenter할지 여부
- upper breach가 composite 기준에서 0.15를 살짝 넘는 문제를 허용할지 여부
- conservative_series를 lower band 고정으로 둘지, line이 lower보다 낮을 때 `min(line, lower)`로 더 보수화할지 여부

## 8. 판정

| 역할 | 후보 | 200티커 판정 |
|---|---|---|
| line_model | PatchTST q25-b2 line_gate | 생존 |
| band_model | CNN-LSTM seq60 q20-b2 + scalar calibration | 생존 |
| composite | PatchTST line + CNN-LSTM calibrated band | 계약상 가능, upper breach와 line_inside_band 정책 보류 |

full 473티커로 바로 가기 전, 다음 CP에서는 composite policy를 명확히 해야 한다. 특히 `line_inside_band`를 단순 기록으로 둘지, band recentering 또는 conservative_series 보정으로 다룰지 결정해야 한다.

## 9. 산출물

- `docs/cp42_200ticker_role_stability_metrics.json`
- `logs/cp42/patchtst_line_200.log`
- `logs/cp42/cnn_lstm_band_200.log`
- `logs/cp42/cnn_lstm_band_200_calibration.json`
- `logs/cp42/composite_200_probe_limited.json`
