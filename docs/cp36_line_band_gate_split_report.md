CP36은 모델 성능 개선이 아니라, 예측선 모델과 밴드 모델을 실제 checkpoint selection 정책에서 분리하는 CP다.

# CP36-M Line/Band Gate Split 보고서

## 1. 목표
기존 `coverage_gate`는 밴드 조건과 예측선 조건을 같이 보는 combined 성격이었다. CP36에서는 예측선 실험과 밴드 실험을 실제 코드 레벨에서 분리하기 위해 `line_gate`, `band_gate`, `combined_gate`를 추가했다.

## 2. Checkpoint Selection 모드
| 모드 | 역할 | 비고 |
|---|---|---|
| `val_total` | 기존 validation forecast loss 기준 | legacy 유지 |
| `line_gate` | 예측선 모델 checkpoint 선택 | coverage 실패와 독립 |
| `band_gate` | 밴드 모델 checkpoint 선택 | IC/spread 실패와 독립 |
| `combined_gate` | line + band 동시 통과 | 기존 coverage gate의 새 이름 |
| `coverage_gate` | deprecated alias | 내부 `gate_type=combined_gate` |

`coverage_gate`는 backward compatibility를 위해 남겼다. 입력값은 유지하지만 실제 gate 성격은 `combined_gate`로 기록한다.

## 3. Gate 기준
`line_gate` eligible 조건은 `spearman_ic > 0`, `long_short_spread > 0`, `mae finite`, `smape finite`다. 정렬은 `spearman_ic desc`, `long_short_spread desc`, `mae asc`, `forecast_loss asc` 순서다.

`band_gate` eligible 조건은 `0.75 <= coverage <= 0.95`, `upper_breach_rate <= 0.15`, `lower_breach_rate <= 0.20`, `avg_band_width > 0`, `band_loss finite`다. 정렬은 `abs(coverage - 0.85) asc`, `upper_breach_rate asc`, `lower_breach_rate asc`, `avg_band_width asc`, `band_loss asc` 순서다.

`combined_gate`는 `line_gate`와 `band_gate`가 모두 통과해야 한다. 정렬은 기존 `coverage_gate`와 동일하게 `upper_breach_rate asc`, `spearman_ic desc`, `long_short_spread desc`, `forecast_loss asc`를 유지했다.

## 4. 저장 메타
checkpoint metrics에 다음 필드를 추가했다.

| 필드 | 의미 |
|---|---|
| `gate_type` | 실제 gate 성격 |
| `gate_failed` | 선택한 gate 실패 여부 |
| `line_gate_pass` | 선택 checkpoint의 line gate 통과 여부 |
| `band_gate_pass` | 선택 checkpoint의 band gate 통과 여부 |
| `combined_gate_pass` | 선택 checkpoint의 combined gate 통과 여부 |
| `role` | `line_model`, `band_model`, `combined_model`, `legacy_val_total` |

기존 호환을 위해 `coverage_gate_failed`도 유지했다. DB 저장 상태는 이제 `selection_result.gate_failed` 기준으로 결정된다. 따라서 `band_gate`가 통과하고 line 지표가 실패한 run도 `role=band_model`로 completed 저장될 수 있다.

## 5. 테스트
다음 테스트를 추가 또는 갱신했다.

| 테스트 범위 | 결과 |
|---|---|
| line_gate pass/fail | 통과 |
| band_gate pass/fail | 통과 |
| combined_gate pass/fail | 통과 |
| coverage_gate alias | 통과 |
| band_gate가 IC 음수여도 band 조건 충족 시 통과 | 통과 |
| line_gate가 coverage 실패여도 line 조건 충족 시 통과 | 통과 |

회귀 명령:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m unittest ai.tests.test_checkpoint_selection ai.tests.test_storage_contracts ai.tests.test_inference_backtest ai.tests.test_patchtst_cli_config
```

결과는 23건 OK다.

## 6. Smoke A: TiDE band_gate
조건은 `tide`, `1D`, `seq_len=252`, `horizon=5`, `q_low=0.10`, `q_high=0.90`, `lambda_band=2.0`, `band_mode=param`, `checkpoint_selection=band_gate`, 50티커, 1epoch다.

| 항목 | 값 |
|---|---:|
| exit code | 0 |
| run_id | tide-1D-0a4a61102c87 |
| selected_reason | band_gate_eligible |
| gate_type | band_gate |
| gate_failed | false |
| line_gate_pass | true |
| band_gate_pass | true |
| combined_gate_pass | true |
| coverage | 0.839193 |
| upper_breach_rate | 0.059264 |
| lower_breach_rate | 0.101543 |
| avg_band_width | 0.255022 |
| spearman_ic | 0.006333 |
| long_short_spread | 0.000257 |
| epoch_seconds | 13.1593 |
| VRAM MB | 69.42 |

## 7. Smoke B: PatchTST line_gate
조건은 `patchtst`, `1D`, `seq_len=252`, `q_low=0.25`, `q_high=0.75`, `lambda_band=2.0`, `band_mode=direct`, `checkpoint_selection=line_gate`, 50티커, 1epoch다.

| 항목 | 값 |
|---|---:|
| exit code | 0 |
| run_id | patchtst-1D-a897bc2553e1 |
| selected_reason | line_gate_eligible |
| gate_type | line_gate |
| gate_failed | false |
| line_gate_pass | true |
| band_gate_pass | false |
| combined_gate_pass | false |
| coverage | 0.988572 |
| upper_breach_rate | 0.006753 |
| lower_breach_rate | 0.004676 |
| avg_band_width | 1.153255 |
| spearman_ic | 0.006389 |
| long_short_spread | 0.001483 |
| epoch_seconds | 78.4229 |
| VRAM MB | 5153.26 |

## 8. 판정
CP36 구현 목적은 충족했다. TiDE smoke에서는 `band_gate`가 정상적으로 밴드 후보를 선택했고, PatchTST smoke에서는 `line_gate`가 coverage 실패와 무관하게 line 후보를 선택했다. 두 smoke 모두 save-run 없이 종료코드 0으로 끝났다.

이번 결과는 성능 채택 판단이 아니다. 다만 다음 단계부터는 band 모델을 line 실패 때문에 버리는 사고를 피할 수 있고, line 모델을 과보수 밴드 때문에 버리는 사고도 피할 수 있다.

## 9. 남은 리스크
`coverage_gate` 이름은 deprecated alias로 남아 있으므로, 향후 문서와 실행 명령은 `combined_gate`를 우선 사용해야 한다. 저장 DB에는 새 메타 필드가 config/metrics JSON에 들어가지만, 별도 DB 컬럼으로 분리한 것은 아니다.
