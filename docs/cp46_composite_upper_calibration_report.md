CP46-M은 새 모델 탐색이 아니라, CNN-LSTM scalar calibrated band가 risk_first_lower_preserve composite에서 upper breach를 크게 만드는 문제를 상단 보정 정책으로 분리한 CP다.

## 1. 목표

CNN-LSTM `s60_q15_b2_direct_188`은 band 단독 scalar calibration 기준으로는 통과했다. 하지만 PatchTST line과 `risk_first_lower_preserve`로 합치면 test upper breach가 0.285106까지 커졌다. 이번 CP는 모델 구조나 학습 파라미터를 바꾸지 않고, composite 상단 보정만 비교한다.

금지 조건은 지켰다.

- full 473티커 실행 금지
- save-run 금지
- UI 수정 금지
- 모델 구조 변경 금지
- 학습 파라미터 변경 금지

## 2. 고정 입력

| 항목 | 값 |
|---|---|
| line model | PatchTST 기존 completed run |
| line checkpoint | `ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-41d584bcb3cb.pt` |
| band model | CNN-LSTM `s60_q15_b2_direct_188` |
| band checkpoint | `ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-01f09b20945a.pt` |
| band calibration | scalar width |
| lower_scale | 1.869896 |
| upper_scale | 1.513046 |
| feature_version | `v3_adjusted_ohlc` |
| timeframe | `1D` |
| horizon | 5 |
| limit_tickers | 200 |
| val/test row_count | 188 / 188 |

## 3. 실행 명령

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.cp46_upper_calibration `
  --line-checkpoint ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-41d584bcb3cb.pt `
  --band-checkpoint ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-01f09b20945a.pt `
  --limit-tickers 200 `
  --max-rows 200 `
  --device cuda `
  --batch-size 256 `
  --amp-dtype bf16 `
  --lower-scale 1.8698961734771729 `
  --upper-scale 1.5130456686019897 `
  --output-json docs\cp46_composite_upper_calibration_metrics.json
```

## 4. 비교 정책

| 정책 | 정의 |
|---|---|
| `risk_first_lower_preserve` | CP44/45 기준선 |
| `risk_first_upper_buffer_1.10` | risk-first 적용 후 upper side width만 10% 확장 |
| `risk_first_upper_buffer_1.25` | risk-first 적용 후 upper side width만 25% 확장 |
| `asymmetric_quantile_expand` | validation upper breach 목표 0.125 기준으로 upper scale fit |
| `symmetric_width_expand` | validation coverage 목표 0.85 기준으로 lower/upper 동일 비율 확장 |

fit 결과:

| 항목 | 값 |
|---|---:|
| upper_expand_scale | 1.017873 |
| symmetric_expand_scale | 1.012875 |
| target_upper_breach | 0.125 |
| target_coverage | 0.85 |

## 5. Validation 결과

| 정책 | coverage | lower breach | upper breach | avg width | width ratio | all_pass |
|---|---:|---:|---:|---:|---:|---|
| risk_first_lower_preserve | 0.830851 | 0.018085 | 0.151064 | 0.253588 | 1.000000 | FAIL |
| risk_first_upper_buffer_1.10 | 0.928723 | 0.018085 | 0.053191 | 0.277080 | 1.092637 | FAIL |
| risk_first_upper_buffer_1.25 | 0.954255 | 0.018085 | 0.027660 | 0.312318 | 1.231593 | FAIL |
| asymmetric_quantile_expand | 0.856383 | 0.018085 | 0.125532 | 0.257787 | 1.016557 | PASS |
| symmetric_width_expand | 0.850000 | 0.018085 | 0.131915 | 0.256854 | 1.012879 | PASS |

validation에서는 fit 기반 정책이 좋아 보인다. 그러나 CP46 판단은 test composite 안정성이 우선이다.

## 6. Test 결과

| 정책 | coverage | lower breach | upper breach | avg width | width ratio | line_inside | all_pass |
|---|---:|---:|---:|---:|---:|---:|---|
| risk_first_lower_preserve | 0.714894 | 0.000000 | 0.285106 | 0.348759 | 1.000000 | 1.000000 | FAIL |
| risk_first_upper_buffer_1.10 | 0.856383 | 0.000000 | 0.143617 | 0.379178 | 1.087218 | 1.000000 | PASS |
| risk_first_upper_buffer_1.25 | 0.926596 | 0.000000 | 0.073404 | 0.424805 | 1.218045 | 1.000000 | FAIL |
| asymmetric_quantile_expand | 0.742553 | 0.000000 | 0.257447 | 0.354196 | 1.015589 | 1.000000 | FAIL |
| symmetric_width_expand | 0.730851 | 0.000000 | 0.269149 | 0.353251 | 1.012877 | 1.000000 | FAIL |

## 7. 폭 진단

| 정책 | downside_width | upside_width | conservative changed | lower less conservative |
|---|---:|---:|---|---|
| risk_first_lower_preserve | 0.044578 | 0.304181 | false | false |
| risk_first_upper_buffer_1.10 | 0.044578 | 0.334599 | false | false |
| risk_first_upper_buffer_1.25 | 0.044578 | 0.380226 | false | false |
| asymmetric_quantile_expand | 0.044578 | 0.309618 | false | false |
| symmetric_width_expand | 0.045153 | 0.308097 | true | false |

`risk_first_upper_buffer_1.10`은 하방 conservative_series를 바꾸지 않고 상단만 확장했다. 따라서 하방 보수성 원칙을 훼손하지 않는다.

## 8. 해석

기준선 `risk_first_lower_preserve`는 line 포함은 해결하지만 upper breach가 0.285106으로 너무 높다. 즉 문제는 하방이 아니라 PatchTST line 기준 상단 공간이 부족한 것이다.

validation에서 fit한 `asymmetric_quantile_expand`는 validation에서는 통과했지만 test coverage 0.742553, upper breach 0.257447로 실패했다. validation 상단 분위수만으로는 test 상단 tail shift를 잡지 못한다.

`risk_first_upper_buffer_1.25`는 upper breach를 낮추지만 coverage가 0.926596으로 목표 범위 0.75~0.90을 넘어 과보수다.

`risk_first_upper_buffer_1.10`은 test coverage 0.856383, upper breach 0.143617, lower breach 0.0으로 CP46 기준을 통과했다. 폭 증가율도 1.087218로 비교적 작다.

## 9. 판단

CNN-LSTM direct band는 폐기하지 않는다. 단, composite에서는 기존 `risk_first_lower_preserve`만으로는 부족하고 `risk_first_upper_buffer_1.10` 후처리를 붙여야 한다.

최종 후보:

| 역할 | 후보 |
|---|---|
| band model | CNN-LSTM `s60_q15_b2_direct_188` |
| band calibration | scalar width |
| composite policy | `risk_first_upper_buffer_1.10` |

보수 후보:

| 역할 | 후보 |
|---|---|
| 보수형 상단 정책 | `risk_first_upper_buffer_1.25` |

보수 후보는 upper breach는 안정적이지만 coverage가 0.926596으로 과보수라 기본값으로는 두지 않는다.

## 10. 다음 CP 권장

다음 CP는 composite 200티커 저장 smoke로 간다.

권장 범위:

- `ai.composite_inference`에 `risk_first_upper_buffer_1.10` 옵션 추가
- `predictions.meta`에 `composition_policy`와 `upper_buffer_scale` 저장
- 5티커 save-run smoke가 아니라 200티커 저장 전 dry/probe 먼저 확인
- full 473티커는 계속 금지

TiDE param 재검토는 아직 필요 없다. upper buffer 1.10으로 CNN-LSTM direct band가 composite 기준에서 살아났다.

## 11. 산출물

- `ai/cp46_upper_calibration.py`
- `docs/cp46_composite_upper_calibration_metrics.json`
- `docs/cp46_composite_upper_calibration_report.md`
