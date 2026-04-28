# CP25-R PatchTST q30/q35 100티커 안정성 확인

## 1. 목적

CP24-R에서 남긴 기본형 `q30-b2`와 공격형 `q35-b2`가 50티커에서만 좋아 보인 것인지 확인했다.

이번 CP는 full run 전 마지막 안정성 smoke이며, 성능 욕심이 아니라 universe 확장 시 밴드/투자 지표가 유지되는지 보는 단계다.

## 2. 실행 조건

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
  --band-mode direct `
  --lambda-band 2.0
```

실행 후보:

| 후보 | q_low | q_high | 종료코드 |
|---|---:|---:|---:|
| q30-b2 | 0.30 | 0.70 | 0 |
| q35-b2 | 0.35 | 0.65 | 0 |

실제 eligible ticker는 100개 입력 중 93개다. `AMP`, `APA`, `BALL`, `BK`, `BXP`, `CHTR`, `CINF`는 재무 sufficiency gate로 제외됐다.

## 3. 산출물

| 항목 | 경로 |
|---|---|
| q30 stdout | `docs/cp25_A_q30_b2_100_1d.stdout.log` |
| q30 stderr | `docs/cp25_A_q30_b2_100_1d.stderr.log` |
| q35 stdout | `docs/cp25_B_q35_b2_100_1d.stdout.log` |
| q35 stderr | `docs/cp25_B_q35_b2_100_1d.stderr.log` |
| 후처리 지표 JSON | `docs/cp25_patchtst_100ticker_stability_metrics.json` |

## 4. 1D 100티커 Validation 결과

| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | median_band_width | p90_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction | epoch_seconds | VRAM peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q30-b2 | 0.463702 | 0.252697 | 0.283601 | 0.069402 | 3.306672 | 0.063201 | 0.101809 | 0.038978 | 0.026855 | 0.001382 | -0.184498 | 0.475238 | 0.370101 | 0.040201 | 149.15 | 5153.27 |
| q35-b2 | 0.384560 | 0.292017 | 0.323423 | 0.056351 | 2.684851 | 0.051311 | 0.082843 | 0.041592 | 0.028621 | 0.001519 | -0.108444 | 0.475388 | 0.371254 | 0.040313 | 145.73 | 5153.27 |

## 5. 1D 100티커 Test 결과

| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | median_band_width | p90_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q30-b2 | 0.445939 | 0.252348 | 0.301713 | 0.073835 | 3.034164 | 0.066307 | 0.111864 | 0.044002 | 0.043314 | 0.006642 | 7.795829 | 0.501301 | 0.378008 | 0.041895 |
| q35-b2 | 0.369709 | 0.293671 | 0.336620 | 0.059944 | 2.463338 | 0.053894 | 0.090991 | 0.046929 | 0.043018 | 0.006612 | 7.737638 | 0.501091 | 0.379001 | 0.042021 |

## 6. CP24 50티커 Validation 기준 대비

| 후보 | 50 coverage | 100 coverage | 50 upper_breach | 100 upper_breach | 50 avg_band_width | 100 avg_band_width | 50 spearman_ic | 100 spearman_ic | 50 long_short_spread | 100 long_short_spread | 50 fee_adjusted_return | 100 fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q30-b2 | 0.890662 | 0.463702 | 0.078626 | 0.283601 | 0.214104 | 0.069402 | 0.069014 | 0.026855 | 0.004678 | 0.001382 | 4.653222 | -0.184498 |
| q35-b2 | 0.837282 | 0.384560 | 0.112994 | 0.323423 | 0.186324 | 0.056351 | 0.068284 | 0.028621 | 0.004552 | 0.001519 | 4.168714 | -0.108444 |

50티커에서 안정적으로 보였던 밴드가 100티커로 확장되면서 지나치게 좁아졌다. 특히 q30-b2도 coverage 0.85~0.93 유지 기준을 크게 벗어났고, q35-b2는 upper breach 0.15 경고선을 두 배 이상 넘었다.

## 7. 판정

q30-b2:

- 기본 후보 유지 기준 실패.
- validation coverage 0.4637은 목표 0.85~0.93과 너무 멀다.
- IC와 spread는 양수지만 50티커 대비 크게 약해졌고, validation fee-adjusted return도 음수다.
- full run 후보로 올리면 안 된다.

q35-b2:

- 공격형 후보 유지 기준 실패.
- validation upper breach 0.3234는 허용 경고선 0.15를 크게 넘는다.
- test 투자 지표는 좋아 보이지만 validation/test 격차가 크므로 안정성 근거로 보기 어렵다.
- 공격형 preset으로 가져가기에는 리스크가 크다.

공통 판단:

- 둘 다 NaN/OOM은 없었지만, calibration 안정성은 무너졌다.
- 50티커 결과는 universe subset 효과가 컸을 가능성이 높다.
- full run은 금지한다.

## 8. Validation/Test 격차

100티커 test에서는 두 후보 모두 IC, long-short spread, fee-adjusted return이 양수로 좋게 나왔다. 하지만 validation에서는 fee-adjusted return이 음수이고 coverage가 크게 붕괴했다.

이 격차는 full run 전 walk-forward 또는 ticker group split 검증이 필요하다는 신호다. 지금 test 성과만 보고 full run으로 가면 과적합/구간 의존 결과를 착각할 수 있다.

## 9. 1W 선택 smoke 결과

선택 항목으로 `q30-b2 1W 50티커`를 시도했지만 학습 전 단계에서 실패했다.

| 항목 | 결과 |
|---|---|
| 명령 | `--timeframe 1W --limit-tickers 50 --epochs 3` |
| 종료코드 | 1 |
| 실패 위치 | `prepare_dataset_splits` / `build_lazy_sequence_dataset` |
| 메시지 | 학습 데이터가 비어 있음. indicators와 price_data 확인 필요 |

q30 1W가 데이터 준비 단계에서 실패했으므로 q35 1W는 진행하지 않았다. 이는 모델 NaN 문제가 아니라 현재 1W 학습 프레임 가용성 문제다.

## 10. 결론

CP25는 q30/q35가 50티커에서만 운 좋게 좋아 보였는지 확인하는 단계였고, 답은 “그럴 가능성이 높다”다.

`q30-b2`와 `q35-b2`는 100티커 validation에서 모두 무너졌으므로 full run 후보로 올리지 않는다.

다음 후보는 CP23/CP24에서 reserve로 남긴 `q25-b2` 100티커 재검증이다. q25-b2도 무너지면 q 범위를 0.20~0.25 쪽으로 되돌리거나, 100티커 기준에서 calibration을 다시 잡아야 한다.

## 11. 다음 CP 제안

1. `q25-b2` 100티커 3epoch 재검증.
2. q25도 실패하면 `q20-b2` 100티커 재검증.
3. 1W는 별도 CP에서 indicators/price_data coverage부터 진단 후 다시 실행.
4. validation/test 격차가 계속 크면 full run 전에 walk-forward 또는 ticker group split을 먼저 붙인다.

## 12. 남은 리스크

- 이번 100티커 결과는 3epoch smoke라 수렴 완료 결과는 아니다. 다만 같은 조건에서 50티커와 비교하는 안정성 smoke로는 충분히 경고 신호다.
- q30/q35가 test에서 좋게 보이는 현상은 validation과 모순되므로, 투자 지표만 보고 후보를 살리면 안 된다.
- 1W 경로는 데이터 준비 단계가 막혀 있어 모델 비교 이전에 데이터 커버리지 점검이 필요하다.
