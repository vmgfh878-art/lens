# CP23-R PatchTST Band Calibration Narrow Smoke

## 1. 실행 환경

- 날짜: 2026-04-28
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- Python: `C:\Users\user\lens\.venv\Scripts\python.exe`
- 공통 조건: `--model patchtst --timeframe 1D --seq-len 252 --epochs 3 --batch-size 256 --device cuda --no-wandb --no-compile --ci-aggregate target --line-target-type raw_future_return --band-target-type raw_future_return --limit-tickers 50 --patch-len 16 --patch-stride 8 --band-mode direct`
- 금지 조건 확인: full 473티커 미실행, `--save-run` 미사용, W&B 미사용, TiDE/CNN-LSTM 미수정, direction/rank head 미추가, width penalty 미복구, 모델 구조 미변경

## 2. 기준점

CP22-R의 기준점은 `q20-b2`다.

| metric | 값 |
|---|---:|
| q_low / q_high | 0.20 / 0.80 |
| lambda_band | 2.0 |
| coverage | 0.972607 |
| avg_band_width | 0.303344 |
| spearman_ic | 0.069890 |
| long_short_spread | 0.005254 |
| fee_adjusted_return | 6.956047 |

## 3. 결과표

| 조건 | q_low | q_high | lambda_band | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction | epoch_seconds | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A q20-b2 | 0.20 | 0.80 | 2.0 | 0.972607 | 0.006906 | 0.020486 | 0.303344 | 0.061271 | 0.069890 | 0.005254 | 6.956047 | 0.468342 | 0.253601 | 0.041565 | 74.4961 | 7783 MB |
| B q25-b2 | 0.25 | 0.75 | 2.0 | 0.936846 | 0.016807 | 0.046347 | 0.250114 | 0.064190 | 0.069396 | 0.005063 | 6.115722 | 0.465266 | 0.246405 | 0.041484 | 75.3191 | 7635 MB |
| C q30-b2 | 0.30 | 0.70 | 2.0 | 0.890662 | 0.030712 | 0.078626 | 0.214104 | 0.067403 | 0.069014 | 0.004678 | 4.653222 | 0.463699 | 0.241007 | 0.041392 | 75.2588 | 7636 MB |
| D q25-b3 | 0.25 | 0.75 | 3.0 | 0.948210 | 0.014636 | 0.037154 | 0.263623 | 0.067209 | 0.066624 | 0.004731 | 4.723813 | 0.459347 | 0.222703 | 0.040814 | 75.3853 | 7635 MB |
| E q30-b3 | 0.30 | 0.70 | 3.0 | 0.921479 | 0.022738 | 0.055783 | 0.237363 | 0.073370 | 0.064092 | 0.003857 | 2.291304 | 0.460623 | 0.210911 | 0.040767 | 75.3171 | 7634 MB |
| F q35-b2 | 0.35 | 0.65 | 2.0 | 0.837282 | 0.049724 | 0.112994 | 0.186324 | 0.070265 | 0.068284 | 0.004552 | 4.168714 | 0.464802 | 0.235889 | 0.041396 | 75.2610 | 7634 MB |

## 4. Coverage 판단

- A는 0.9726으로 아직 관찰 가능 상단이다.
- B는 0.9368로 현실적 후보 구간이다.
- C는 0.8907로 이상적 후보 구간 상단에 들어왔다.
- F는 0.8373으로 이상적 구간에 들어왔지만 upper breach가 11.3%까지 커졌다.

## 5. 투자 지표 판단

- B q25-b2는 투자 지표를 가장 안정적으로 보존했다. IC 0.0694, spread 0.0051, fee-adjusted return 6.12로 기준점 대비 하락폭이 작다.
- C q30-b2는 coverage를 목표 근처까지 낮추면서 IC 0.0690과 양수 spread를 유지했다. return은 4.65로 낮아졌지만 무너지진 않았다.
- F q35-b2는 coverage는 가장 좋지만 upper breach와 return 저하가 있어 바로 채택하기 어렵다.
- lambda_band 3.0 계열은 과대예측 지표는 좋아졌지만 투자 지표가 더 약해져 우선순위를 낮춘다.

## 6. 후보 결정

다음 full run 후보는 아직 단일 확정하지 않는다.

1차 후보: `q30-b2`

- q_low=0.30, q_high=0.70, lambda_band=2.0
- coverage가 0.8907로 목표 상단에 들어왔다.
- 밴드 폭은 0.2141로 기준점 대비 줄었다.
- IC와 spread는 유지됐고 return도 양수다.

보수 후보: `q25-b2`

- q_low=0.25, q_high=0.75, lambda_band=2.0
- coverage 0.9368로 아직 높지만 투자 지표 보존이 가장 좋다.

탈락 또는 보류:

- `q35-b2`: coverage는 좋지만 upper breach가 크고 return이 낮다.
- `q25-b3`, `q30-b3`: lambda_band 3.0은 투자 지표가 약해진다.

## 7. 리스크

- 모든 결과는 50티커 3epoch smoke 기준이다.
- upper breach가 quantile을 좁힐수록 빠르게 커진다.
- q30-b2를 full 후보로 올리기 전에 100티커 또는 1W 작은 smoke로 한 번 더 확인하는 편이 안전하다.

## 8. 다음 CP 추천

다음은 full run 직행보다 후보 안정성 확인이 낫다.

- `q25-b2`와 `q30-b2`를 100티커 3epoch로 비교
- 가능하면 1W 50티커 3epoch도 확인
- 둘 중 하나가 유지되면 그때 전체 473티커 5epoch 후보로 올린다.

## 9. 로그 파일

- `docs/cp23_A_q20_b2.stdout.log`
- `docs/cp23_B_q25_b2.stdout.log`
- `docs/cp23_C_q30_b2.stdout.log`
- `docs/cp23_D_q25_b3.stdout.log`
- `docs/cp23_E_q30_b3.stdout.log`
- `docs/cp23_F_q35_b2.stdout.log`
