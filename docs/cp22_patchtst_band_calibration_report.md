# CP22-R PatchTST Band Calibration Smoke

## 1. 실행 환경

- 날짜: 2026-04-28
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- Python: `C:\Users\user\lens\.venv\Scripts\python.exe`
- 공통 조건: `--model patchtst --timeframe 1D --seq-len 252 --epochs 3 --batch-size 256 --device cuda --no-wandb --no-compile --ci-aggregate target --line-target-type raw_future_return --band-target-type raw_future_return --limit-tickers 50 --patch-len 16 --patch-stride 8`
- 금지 조건 확인: full 473티커 미실행, TiDE/CNN-LSTM 미수정, direction/rank head 미추가, width penalty 미복구, `--save-run` 미사용, W&B 미사용

## 2. 지표 추가

`ai/evaluation.py`에 `lower_breach_rate`, `upper_breach_rate`를 추가했다. 두 지표는 band target이 각각 lower band 아래/upper band 위로 벗어난 비율이다.

## 3. 결과표

| 조건 | q_low | q_high | lambda_band | band_mode | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction | epoch_seconds | VRAM peak |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 0.10 | 0.90 | 1.0 | direct | 0.999954 | 0.000023 | 0.000023 | 0.661335 | 0.065957 | 0.069970 | 0.005070 | 6.002602 | 0.458476 | 0.226232 | 0.040160 | 75.2069 | 7848 MB |
| B q15 | 0.15 | 0.85 | 1.0 | direct | 0.998004 | 0.000801 | 0.001196 | 0.465148 | 0.069646 | 0.069701 | 0.004849 | 5.183134 | 0.460333 | 0.237699 | 0.040586 | 75.7110 | 7641 MB |
| C q20 | 0.20 | 0.80 | 1.0 | direct | 0.987256 | 0.003981 | 0.008763 | 0.349639 | 0.070110 | 0.070359 | 0.005125 | 6.504665 | 0.465092 | 0.251048 | 0.041202 | 76.5408 | 7769 MB |
| D q15-b2 | 0.15 | 0.85 | 2.0 | direct | 0.995148 | 0.001439 | 0.003412 | 0.401907 | 0.060262 | 0.069561 | 0.004955 | 5.584883 | 0.463757 | 0.242876 | 0.041161 | 75.1296 | 7768 MB |
| E q20-b2 | 0.20 | 0.80 | 2.0 | direct | 0.972607 | 0.006906 | 0.020486 | 0.303344 | 0.061271 | 0.069890 | 0.005254 | 6.956047 | 0.468342 | 0.253601 | 0.041565 | 75.3091 | 7769 MB |
| F param | 0.15 | 0.85 | 2.0 | param | 0.993976 | 0.002136 | 0.003888 | 0.384076 | 0.057617 | 0.051032 | 0.002202 | 0.201084 | 0.443213 | 0.131751 | 0.036987 | 75.1570 | 7769 MB |

## 4. Band Guardrail

- A/B/D/F는 coverage가 0.98 이상이라 여전히 경고 구간이다.
- C는 coverage 0.9873으로 밴드는 줄었지만 아직 경고 구간이다.
- E는 coverage 0.9726으로 가장 많이 내려왔고, 관찰 가능 구간 상단이다. 아직 0.75~0.90 1차 후보 기준에는 못 들어왔다.
- E는 baseline 대비 avg_band_width를 0.6613에서 0.3033으로 줄였다.

## 5. 투자 지표 판단

- E q20-b2는 `spearman_ic=0.069890`, `long_short_spread=0.005254`, `fee_adjusted_return=6.956047`로 baseline과 같은 수준이거나 더 좋다.
- C q20도 투자 지표는 좋지만 band_loss가 baseline보다 높고 coverage가 더 높다.
- F param은 band_loss와 overprediction 지표는 좋지만 IC, spread, fee-adjusted return이 크게 약해져 탈락이다.

## 6. Breach 해석

- E q20-b2의 breach는 lower 0.69%, upper 2.05%로 upper breach 쪽이 더 많다.
- 즉 밴드를 줄이면서 실제 수익률이 upper band를 넘는 경우가 늘었다.
- 이 비대칭은 다음 calibration에서 q_high 또는 upper-band 학습 쪽을 별도로 점검할 근거다.

## 7. 후보 결정

주력 후보는 아직 확정하지 않는다. 다만 다음 smoke의 기준점은 E q20-b2로 둔다.

- geometry: `patch_len=16`, `patch_stride=8`
- q: `q_low=0.20`, `q_high=0.80`
- loss: `lambda_band=2.0`
- band_mode: `direct`

이 후보는 밴드를 크게 줄이면서 투자 지표를 유지했지만, coverage가 아직 0.9726이라 최종 calibration 후보는 아니다.

## 8. 다음 CP 추천

coverage를 0.75~0.90으로 더 내리려면 q 범위를 더 좁히는 smoke가 필요하다.

추천 매트릭스:

- `q_low=0.25`, `q_high=0.75`, `lambda_band=2.0`, `direct`
- `q_low=0.30`, `q_high=0.70`, `lambda_band=2.0`, `direct`
- `q_low=0.25`, `q_high=0.75`, `lambda_band=3.0`, `direct`
- upper breach가 늘어나는 점을 보려면 `q_low=0.20`, `q_high=0.85`도 보조로 확인

## 9. 로그 파일

- `docs/cp22_A_baseline.stdout.log`
- `docs/cp22_B_q15.stdout.log`
- `docs/cp22_C_q20.stdout.log`
- `docs/cp22_D_q15_b2.stdout.log`
- `docs/cp22_E_q20_b2.stdout.log`
- `docs/cp22_F_param.stdout.log`
