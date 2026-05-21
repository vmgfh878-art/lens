CP43-M은 full run이 아니라, CP42의 PatchTST line + CNN-LSTM calibrated band 조합에서 line_inside_band 문제를 해결할 정책을 비교한 CP다.

## 1. 목표

CP42에서 composite coverage는 0.836170으로 괜찮았지만 `line_inside_band_ratio`가 0.515957에 그쳤고, upper breach도 0.156383으로 기준 0.15를 살짝 넘었다. 이번 CP는 모델 재학습 없이 composite 후처리 정책만 비교한다.

금지 조건은 유지했다.

- full 473티커 실행 금지
- save-run 금지
- W&B 금지
- 모델 구조 변경 금지

## 2. 실행 조건

| 항목 | 값 |
|---|---|
| split | test |
| limit_tickers | 200 |
| row_count | 188 |
| line checkpoint | `ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-d521eff215b1.pt` |
| band checkpoint | `ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-5a347fab1538.pt` |
| band calibration | scalar width |
| lower_scale | 2.731744 |
| upper_scale | 1.767985 |
| device | cuda |
| amp_dtype | bf16 |

실행 명령:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.composite_policy_eval `
  --line-checkpoint ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-d521eff215b1.pt `
  --band-checkpoint ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-5a347fab1538.pt `
  --split test `
  --limit-tickers 200 `
  --max-rows 200 `
  --device cuda `
  --batch-size 256 `
  --amp-dtype bf16 `
  --lower-scale 2.7317440509796143 `
  --upper-scale 1.7679849863052368 `
  --output-json docs\cp43_composite_policy_comparison_metrics.json
```

## 3. 비교 정책

| 정책 | 정의 | 목적 |
|---|---|---|
| raw_composite | CP42 방식 그대로 | 기준선 |
| include_line_clamp | `lower=min(lower,line)`, `upper=max(upper,line)` | line을 항상 밴드 안에 포함 |
| line_centered_asymmetric | `lower=line-downside_width`, `upper=line+upside_width` | CNN-LSTM 폭은 유지하고 중심을 PatchTST line으로 이동 |
| risk_first_lower_preserve | 하방은 더 보수적인 쪽, 상방은 line 포함 | 투자 원칙상 하방 보수성 우선 |

`risk_first_lower_preserve`는 현재 long 기준 raw return 밴드에서는 수치상 `include_line_clamp`와 동일하다. 하방의 “더 보수적인 쪽”이 `min(lower,line)`이고, 상방 line 포함이 `max(upper,line)`이기 때문이다. 차이는 정책 의미와 저장 메타에 남길 이름이다.

## 4. 결과 표

| 정책 | line_inside | coverage | lower_breach | upper_breach | avg_width | width_ratio | IC | long_short | fee_return | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_composite | 0.515957 | 0.836170 | 0.007447 | 0.156383 | 0.286390 | 1.0000 | 0.298836 | 0.025163 | 0.094530 | 탈락 |
| include_line_clamp | 1.000000 | 0.870213 | 0.007447 | 0.122340 | 0.294814 | 1.0294 | 0.298836 | 0.025163 | 0.094530 | 통과 |
| line_centered_asymmetric | 1.000000 | 0.374468 | 0.062766 | 0.562766 | 0.286390 | 1.0000 | 0.298836 | 0.025163 | 0.094530 | 탈락 |
| risk_first_lower_preserve | 1.000000 | 0.870213 | 0.007447 | 0.122340 | 0.294814 | 1.0294 | 0.298836 | 0.025163 | 0.094530 | 통과 |

## 5. 기준별 판정

| 기준 | raw | clamp | centered | risk-first |
|---|---|---|---|---|
| line_inside_band_ratio >= 0.95 | FAIL | PASS | PASS | PASS |
| coverage 0.75~0.90 | PASS | PASS | FAIL | PASS |
| lower_breach <= 0.12 | PASS | PASS | PASS | PASS |
| upper_breach <= 0.15 | FAIL | PASS | FAIL | PASS |
| width 과도 증가 없음 | PASS | PASS | PASS | PASS |

## 6. 해석

raw composite는 coverage 자체는 좋지만 line이 밴드 밖에 있는 행이 절반 가까이 발생한다. upper breach도 0.15를 넘어서 CP43 기준을 통과하지 못한다.

include-line clamp는 line_inside를 1.0으로 만들면서 coverage를 0.870213으로 유지했고, upper breach를 0.122340까지 낮췄다. avg_band_width는 raw 대비 2.94%만 증가해서 인위적 확대가 과도하다고 보기 어렵다.

line-centered asymmetric은 직관과 달리 실패했다. CNN-LSTM의 calibrated downside/upside 폭을 PatchTST line 중심으로 그대로 옮기면 실제 target이 upper를 넘는 비율이 0.562766까지 치솟는다. 즉 두 모델의 중심선 분포가 달라서 단순 recentering은 안전하지 않다.

risk-first lower preserve는 현재 수식상 clamp와 같은 결과다. 다만 정책 이름은 `risk_first_lower_preserve`를 채택하는 편이 낫다. 이유는 conservative_series를 하방 위험 우선으로 해석하는 프로젝트 원칙과 더 잘 맞기 때문이다.

## 7. 권고

다음 composite 기본 정책은 `risk_first_lower_preserve`로 둔다.

- line_inside_band_ratio: 1.000000
- coverage: 0.870213
- lower_breach_rate: 0.007447
- upper_breach_rate: 0.122340
- avg_band_width 증가: raw 대비 2.94%

UI나 저장 메타에는 `composition_policy=risk_first_lower_preserve`를 남기는 것이 좋다. 수치상 clamp와 같아도, 투자 관점에서는 “line을 억지로 밴드 안에 넣었다”보다 “하방 보수성을 우선했다”가 더 정확한 설명이다.

## 8. 다음 CP 제안

CP44에서는 CP43 정책을 `ai.composite_inference` 저장 경로에 옵션으로 연결한다.

제안 범위:

- `--composition-policy raw_composite|include_line_clamp|line_centered_asymmetric|risk_first_lower_preserve` 추가
- 기본값은 당장 바꾸지 말고 `risk_first_lower_preserve`를 명시 실행으로 검증
- 5티커 save-run smoke로 `predictions.meta.composition_policy` 저장 확인
- full 473은 계속 금지

## 9. 산출물

- `ai/composite_policy_eval.py`
- `docs/cp43_composite_policy_comparison_metrics.json`
- `docs/cp43_composite_policy_comparison_report.md`
