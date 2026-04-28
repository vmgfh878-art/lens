# CP26-R PatchTST 100티커 보수 calibration 재확인

## 1. 목적

CP25-R에서 `q30-b2`, `q35-b2`가 100티커 validation에서 무너진 것을 인정하고, 100티커 기준으로 밴드 calibration 후보를 다시 찾았다.

이번 CP에서는 1W 학습을 금지하고 1D 100티커 calibration만 수행했다.

## 2. 사전 조치

CP25-R의 1W 실패로 비워졌던 `ai/cache/ticker_id_map_1w.json`을 기존 tracked 상태로 복원했다.

이번 CP에서는 1W 학습을 재시도하지 않았다.

## 3. 실행 조건

공통 조건:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train `
  --model patchtst `
  --timeframe 1D `
  --seq-len 252 `
  --epochs 3 `
  --batch-size 256 `
  --device cuda `
  --no-wandb `
  --no-compile `
  --ci-aggregate target `
  --line-target-type raw_future_return `
  --band-target-type raw_future_return `
  --limit-tickers 100 `
  --patch-len 16 `
  --patch-stride 8 `
  --band-mode direct
```

실제 eligible ticker는 100개 입력 중 93개다. `AMP`, `APA`, `BALL`, `BK`, `BXP`, `CHTR`, `CINF`는 재무 gate로 제외됐다.

## 4. 실행 후보

| 후보 | q_low | q_high | lambda_band | 종료코드 | run_id |
|---|---:|---:|---:|---:|---|
| q25-b2 | 0.25 | 0.75 | 2.0 | 0 | `patchtst-1D-f88da626c275` |
| q20-b2 | 0.20 | 0.80 | 2.0 | 0 | `patchtst-1D-e370a0dc990b` |
| q15-b2 | 0.15 | 0.85 | 2.0 | 0 | `patchtst-1D-40152e1b9f83` |
| q20-b1 | 0.20 | 0.80 | 1.0 | 0 | `patchtst-1D-06f8713d4a04` |

## 5. 100티커 Validation 최종 checkpoint 결과

| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction | epoch_seconds | VRAM peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.544565 | 0.212487 | 0.242948 | 0.083824 | 3.993848 | 0.035592 | 0.026765 | 0.001303 | -0.206959 | 0.474790 | 0.371954 | 0.040218 | 149.64 | 5153.27 |
| q20-b2 | 0.650025 | 0.161015 | 0.188960 | 0.104741 | 4.990448 | 0.031480 | 0.012186 | 0.001467 | -0.118567 | 0.476015 | 0.367232 | 0.039935 | 149.89 | 5153.27 |
| q15-b2 | 0.749335 | 0.113195 | 0.137470 | 0.128752 | 6.134445 | 0.026372 | 0.015283 | 0.002072 | 0.314993 | 0.476643 | 0.369318 | 0.039953 | 149.79 | 5153.27 |
| q20-b1 | 0.652703 | 0.160537 | 0.186760 | 0.105156 | 5.010210 | 0.031486 | 0.015393 | 0.001486 | -0.094951 | 0.474192 | 0.364841 | 0.039563 | 149.59 | 5153.27 |

## 6. 100티커 Test 최종 checkpoint 결과

| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.523747 | 0.210715 | 0.265539 | 0.089183 | 3.664863 | 0.040211 | 0.043502 | 0.006891 | 8.626742 | 0.501301 | 0.379916 | 0.041935 |
| q20-b2 | 0.624928 | 0.160447 | 0.214625 | 0.111446 | 4.579756 | 0.035493 | 0.043478 | 0.007037 | 9.236534 | 0.502825 | 0.374965 | 0.041652 |
| q15-b2 | 0.720787 | 0.114909 | 0.164304 | 0.137009 | 5.630221 | 0.029647 | 0.043953 | 0.007058 | 9.340334 | 0.503304 | 0.377189 | 0.041657 |
| q20-b1 | 0.627403 | 0.159879 | 0.212718 | 0.111912 | 4.598902 | 0.035500 | 0.044802 | 0.007117 | 9.566030 | 0.500344 | 0.373171 | 0.041240 |

## 7. CP22/23 50티커 대비 100티커 최종 결과

| 후보 | 50 coverage | 100 coverage | 50 upper_breach | 100 upper_breach | 50 avg_band_width | 100 avg_band_width | 50 IC | 100 IC | 50 spread | 100 spread | 50 fee_return | 100 fee_return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.936846 | 0.544565 | 0.046347 | 0.242948 | 0.250114 | 0.083824 | 0.069396 | 0.026765 | 0.005063 | 0.001303 | 6.115722 | -0.206959 |
| q20-b2 | 0.972607 | 0.650025 | 0.020486 | 0.188960 | 0.303344 | 0.104741 | 0.069890 | 0.012186 | 0.005254 | 0.001467 | 6.956047 | -0.118567 |
| q15-b2 | 0.995148 | 0.749335 | 0.003727 | 0.137470 | 0.401907 | 0.128752 | 0.069561 | 0.015283 | 0.004955 | 0.002072 | 5.584883 | 0.314993 |
| q20-b1 | 0.987256 | 0.652703 | 0.008171 | 0.186760 | 0.349639 | 0.105156 | 0.070359 | 0.015393 | 0.005125 | 0.001486 | 6.504665 | -0.094951 |

100티커로 확장하면 전반적으로 밴드 폭과 coverage가 크게 줄고, IC/spread도 약해진다. 50티커 결과는 calibration 후보를 바로 full run으로 올리기에는 universe 편향이 컸다.

## 8. Epoch trajectory

최종 checkpoint는 validation total loss 기준으로 저장된다. 그런데 이번 실험에서는 epoch가 진행될수록 total loss는 좋아지지만 밴드가 계속 좁아져 coverage가 무너지는 패턴이 보였다.

| 후보 | epoch | coverage | upper_breach_rate | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 1 | 0.889698 | 0.081341 | 0.222528 | 0.059091 | 0.024434 | 0.002503 | 0.479462 |
| q25-b2 | 2 | 0.598757 | 0.201225 | 0.094728 | 0.036309 | 0.029348 | 0.001559 | -0.081611 |
| q25-b2 | 3 | 0.544565 | 0.242948 | 0.083824 | 0.035592 | 0.026765 | 0.001303 | -0.206959 |
| q20-b2 | 1 | 0.924373 | 0.053875 | 0.248139 | 0.051846 | 0.024377 | 0.002717 | 0.793004 |
| q20-b2 | 2 | 0.697080 | 0.153436 | 0.115872 | 0.032131 | 0.015843 | 0.002471 | 0.704743 |
| q20-b2 | 3 | 0.650025 | 0.188960 | 0.104741 | 0.031480 | 0.012186 | 0.001467 | -0.118567 |
| q15-b2 | 1 | 0.968015 | 0.024418 | 0.296995 | 0.045399 | 0.024582 | 0.002424 | 0.428716 |
| q15-b2 | 2 | 0.802552 | 0.095257 | 0.144704 | 0.027121 | 0.018834 | 0.002331 | 0.545685 |
| q15-b2 | 3 | 0.749335 | 0.137470 | 0.128752 | 0.026372 | 0.015283 | 0.002072 | 0.314993 |
| q20-b1 | 1 | 0.970286 | 0.019797 | 0.309007 | 0.062595 | 0.012842 | 0.002326 | 0.365862 |
| q20-b1 | 2 | 0.696393 | 0.149413 | 0.115530 | 0.032063 | 0.018904 | 0.002113 | 0.347574 |
| q20-b1 | 3 | 0.652703 | 0.186760 | 0.105156 | 0.031486 | 0.015393 | 0.001486 | -0.094951 |

이 표가 이번 CP의 제일 중요한 진단이다. `val_total` 기준 best와 coverage guardrail 기준 best가 다르다.

## 9. 판정

q25-b2:

- 최종 checkpoint 기준 coverage 0.5446으로 탈락.
- epoch1은 coverage 0.8897로 후보권이지만 현재 저장된 checkpoint가 아니다.

q20-b2:

- 최종 checkpoint 기준 coverage 0.6500으로 탈락.
- epoch1은 coverage 0.9244로 후보권, epoch2는 0.6971로 탈락 경계다.

q15-b2:

- 최종 checkpoint 기준 coverage 0.7493으로 공격 후보.
- upper breach 0.1375로 0.15 경고선 아래다.
- IC와 spread는 양수이고 fee_adjusted_return도 양수다.
- 다만 기본 후보가 아니라 공격 후보이며, test upper breach 0.1643은 경고선보다 높다.

q20-b1:

- 최종 checkpoint 기준 coverage 0.6527로 탈락.
- lambda_band 1.0이 q20-b2보다 나은 기본 후보를 만들지는 못했다.

## 10. 결론

100티커 최종 checkpoint 기준으로 기본 후보는 아직 없다.

살아남은 것은 `q15-b2` 공격 후보뿐이다. validation coverage 0.7493, upper breach 0.1375, IC 0.0153, spread 0.0021, fee_adjusted_return 0.315로 최소 guardrail은 통과한다.

하지만 full 473티커로 바로 가면 안 된다. 이유는 현재 학습 루프가 val_total 기준으로 checkpoint를 고르면서 coverage를 과도하게 희생하는 경향이 확인됐기 때문이다.

## 11. 다음 CP 제안

1. coverage-aware checkpoint selection 추가 또는 별도 evaluation-only selector 작성.
2. q15-b2를 epoch2 기준으로 재현하기 위해 `epochs=2` smoke를 한 번 더 실행.
3. q20-b2/q25-b2도 epoch1이 후보권이므로, 짧은 epoch 또는 coverage monitor 기준으로 재평가.
4. full run은 coverage-aware selection이 닫히기 전까지 금지.

## 12. 남은 리스크

- 3epoch smoke에서 epoch별 calibration 변화가 크다. 단순 final checkpoint 비교만으로는 후보를 과소/과대평가할 수 있다.
- validation/test 투자 지표 격차가 여전히 크다. full run 전 walk-forward 또는 ticker group split이 필요할 수 있다.
- q15-b2는 공격 후보일 뿐 기본 후보가 아니다. 기본형 preset은 아직 찾지 못했다.
