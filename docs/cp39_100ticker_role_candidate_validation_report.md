CP39는 full run이 아니라, line 후보 PatchTST와 band 후보 CNN-LSTM의 100티커 역할별 안정성 검증 CP다.

# CP39-M 100티커 역할 후보 검증 보고서

## 1. 목표
CP38에서 생존한 `CNN-LSTM seq60 q20-b2 direct + scalar width calibration` 밴드 후보를 100티커에서 검증했다. 동시에 PatchTST line 후보가 같은 계약에서 조합 가능한지 최소 수준으로 확인했다.

## 2. 공통 조건
공통 조건은 `feature_version=v3_adjusted_ohlc`, `timeframe=1D`, `horizon=5`, `batch_size=256`, `--no-compile`, `--no-wandb`, `--save-run` 미사용, 100티커 3epoch다. full 473티커, 신규 모델, DLinear/NLinear, 시그널 모델화는 하지 않았다.

## 3. CNN-LSTM Band 후보
조건은 `model=cnn_lstm`, `seq_len=60`, `q_low=0.20`, `q_high=0.80`, `lambda_band=2.0`, `band_mode=direct`, `checkpoint_selection=band_gate`, `--fp32-modules lstm,heads`다.

| 항목 | 값 |
|---|---:|
| exit code | 0 |
| run_id | cnn_lstm-1D-b882658d1561 |
| selected_epoch | 3 |
| selected_reason | band_gate_failed_fallback_val_total |
| raw band_gate_pass | false |
| 평균 epoch seconds | 52.37 |
| VRAM peak MB | 323.93 |

원본 band는 100티커에서 실패했다.

| split | coverage | lower breach | upper breach | avg band width | band loss |
|---|---:|---:|---:|---:|---:|
| validation | 0.672702 | 0.175678 | 0.151621 | 0.048305 | 0.015704 |
| test | 0.573707 | 0.264400 | 0.161893 | 0.053179 | 0.029042 |

## 4. Scalar Width Calibration
validation 기준으로 scalar width calibration을 fit했다.

| 계수 | 값 |
|---|---:|
| lower scale | 1.908173 |
| upper scale | 1.378499 |

적용 결과:

| split | coverage | lower breach | upper breach | avg band width | band loss |
|---|---:|---:|---:|---:|---:|
| validation | 0.850022 | 0.074968 | 0.075010 | 0.087036 | 0.021353 |
| test | 0.796509 | 0.112616 | 0.090876 | 0.098413 | 0.024504 |

판정 기준인 test coverage 0.75~0.95, upper breach <=0.15, lower breach <=0.20을 모두 통과했다. avg_band_width는 원본 대비 늘었지만 폭증으로 보기는 어렵다.

## 5. Conformal Residual 비교
conformal residual calibration도 비교했지만 생존하지 못했다.

| 방식 | test coverage | lower breach | upper breach | avg band width | 판정 |
|---|---:|---:|---:|---:|---|
| scalar width | 0.796509 | 0.112616 | 0.090876 | 0.098413 | 생존 |
| conformal residual | 0.731998 | 0.210933 | 0.057069 | 0.079100 | 탈락 |

따라서 CP39 band 후보는 scalar width calibration만 유지한다.

## 6. PatchTST Line 후보
조건은 `model=patchtst`, `seq_len=252`, `patch_len=16`, `patch_stride=8`, `q_low=0.25`, `q_high=0.75`, `lambda_band=2.0`, `checkpoint_selection=line_gate`다.

| 항목 | 값 |
|---|---:|
| exit code | 0 |
| run_id | patchtst-1D-19103d294e6b |
| selected_epoch | 2 |
| selected_reason | line_gate_eligible |
| line_gate_pass | true |
| 평균 epoch seconds | 155.46 |
| VRAM peak MB | 5153.27 |

| split | spearman_ic | long_short_spread | mae | smape | direction_accuracy |
|---|---:|---:|---:|---:|---:|
| validation | 0.018231 | 0.002256 | 0.049054 | 1.525516 | 0.476232 |
| test | 0.045832 | 0.006690 | 0.055652 | 1.508459 | 0.504289 |

PatchTST는 100티커에서도 line 후보 기준을 통과했다.

## 7. 조합 가능성
PatchTST line과 CNN-LSTM calibrated band는 조합 가능하다.

공통 계약:

| 항목 | 값 |
|---|---|
| feature_version | `v3_adjusted_ohlc` |
| timeframe | `1D` |
| horizon | `5` |
| line_target_type | `raw_future_return` |
| band_target_type | `raw_future_return` |

seq_len은 다르다. PatchTST는 252, CNN-LSTM은 60이다. 하지만 출력 계약은 같은 ticker/asof_date/horizon의 raw return 공간이므로 inference 단계에서 조합 가능하다.

## 8. 저장 구조 초안
실제 결합 구현은 하지 않았다. 다만 저장 구조는 나중에 분리해야 한다.

필요 필드 초안:

- `line_model_run_id`
- `band_model_run_id`
- `band_calibration_method`
- `band_calibration_params`
- `line_series_source`
- `band_series_source`

현재 `predictions.run_id` 하나로 line/band 출처를 동시에 표현하면 역할 분리 모델을 제대로 추적하기 어렵다. 조합 run을 별도 composite run으로 둘지, predictions에 line/band run id를 추가할지 다음 CP에서 결정해야 한다.

## 9. Conservative Series 초안
long-only 관점에서는 다음 정의가 가장 단순하고 보수적이다.

- `line_series`: PatchTST line output
- `lower_band_series`: calibrated CNN-LSTM lower band
- `upper_band_series`: calibrated CNN-LSTM upper band
- `conservative_series`: `calibrated_lower_band_series`

short 관점까지 열 경우에는 `conservative_short_series = calibrated_upper_band_series`를 별도 정의하는 편이 안전하다. 하나의 `conservative_series`에 long/short 의미를 섞으면 해석이 흐려진다.

## 10. 판정
CP39 기준 역할 후보는 다음처럼 정리한다.

| 역할 | 후보 | 상태 |
|---|---|---|
| line_model | PatchTST q25-b2 line_gate | 생존 |
| band_model | CNN-LSTM seq60 q20-b2 direct + scalar width calibration | 생존 |

full 473티커는 아직 금지다. 다음은 200티커 제한 검증 또는 저장 계약 확장 스모크 중 하나를 선택해야 한다.
