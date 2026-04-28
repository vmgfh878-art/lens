# CP24-R Bollinger Baseline vs PatchTST Band Calibration

## 1. 목적

CP23-R에서 남긴 PatchTST 밴드 후보 `q25-b2`, `q30-b2`, `q35-b2`를 전통적 Bollinger 기준선과 같은 validation/test split에서 비교했다.

핵심 질문은 `q35-b2`의 validation upper breach 11.30%가 과하게 좁은 밴드인지, 아니면 전통 밴드 기준으로 허용 가능한 공격형 후보인지다.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| 작업 | 기존 CP23 체크포인트 재평가 + Bollinger 기준선 계산 |
| 신규 학습 | 없음 |
| device | CUDA |
| model | PatchTST |
| timeframe | 1D |
| seq_len | 252 |
| horizon | 5 |
| batch_size | 256 |
| limit_tickers | 50 입력, 48 eligible |
| split samples | train 80459 / validation 17231 / test 17171 |
| 저장 정책 | `--save-run` 없음, W&B 없음 |

제외 ticker는 기존 split과 동일하게 `AMP`, `APA`가 재무 sufficiency gate로 빠졌다.

## 3. 비교 방법

Bollinger는 가격 밴드가 아니라 PatchTST와 같은 `raw_future_return` 공간에서 계산했다.

각 sample의 as-of index를 `t`, horizon을 `h=1..5`로 둘 때, 과거에 이미 완료된 누적수익률 `close[j+h] / close[j] - 1`만 사용했다. 조건은 `j + h <= t`이며, target 구간은 rolling 통계에 들어가지 않는다.

각 Bollinger 밴드는 horizon별로 최근 `N`개 완료 수익률의 평균과 표준편차를 이용해 `mean ± k * std`로 만들었다. 표준편차는 `ddof=0`을 사용했다.

`relative_band_width`는 `avg_band_width / mean(abs(raw_future_return))`로 계산했다.

## 4. PatchTST 후보 Validation Band 지표

| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | median_band_width | p90_band_width | band_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.936846 | 0.016807 | 0.046347 | 0.250114 | 11.5155 | 0.227793 | 0.382816 | 0.064190 |
| q30-b2 | 0.890662 | 0.030712 | 0.078626 | 0.214104 | 9.8576 | 0.194878 | 0.329700 | 0.067403 |
| q35-b2 | 0.837282 | 0.049724 | 0.112994 | 0.186324 | 8.5786 | 0.169717 | 0.290268 | 0.070265 |

## 5. PatchTST 후보 Validation 투자 지표

| 후보 | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy | overprediction_rate | mean_overprediction |
|---|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.069396 | 0.005063 | 6.115722 | 0.465266 | 0.246405 | 0.041484 |
| q30-b2 | 0.069014 | 0.004678 | 4.653222 | 0.463699 | 0.241007 | 0.041392 |
| q35-b2 | 0.068284 | 0.004552 | 4.168714 | 0.464802 | 0.235889 | 0.041396 |

투자 지표는 q를 좁힐수록 조금씩 약해지지만, q35-b2까지도 IC와 long-short spread는 양수로 유지됐다.

## 6. Bollinger Validation 기준선

| 기준선 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | median_band_width | p90_band_width |
|---|---:|---:|---:|---:|---:|---:|---:|
| BB20-2.0s | 0.879566 | 0.061668 | 0.058766 | 0.102315 | 4.7107 | 0.086865 | 0.172918 |
| BB20-1.5s | 0.788660 | 0.105937 | 0.105403 | 0.076736 | 3.5330 | 0.065148 | 0.129688 |
| BB20-1.0s | 0.622274 | 0.188486 | 0.189240 | 0.051158 | 2.3553 | 0.043432 | 0.086459 |
| BB60-2.0s | 0.930822 | 0.034589 | 0.034589 | 0.111930 | 5.1534 | 0.099704 | 0.180210 |
| BB60-1.5s | 0.857153 | 0.070559 | 0.072288 | 0.083947 | 3.8650 | 0.074778 | 0.135158 |

## 7. Validation 비교 판단

`q25-b2`는 coverage 0.9368로 BB60-2.0s의 0.9308과 비슷하다. 다만 avg_band_width는 0.2501로 BB60-2.0s의 0.1119보다 2.2배 넓다. 보수형 후보로 둘 수 있지만, Bollinger 대비 폭은 아직 넓다.

`q30-b2`는 coverage 0.8907로 BB20-2.0s의 0.8796과 비슷하고, upper breach 7.86%는 BB20-2.0s 5.88%와 BB20-1.5s 10.54% 사이에 있다. 기본 후보로 가장 균형이 좋다.

`q35-b2`는 coverage 0.8373으로 BB60-1.5s 0.8572와 BB20-1.5s 0.7887 사이에 있고, upper breach 11.30%는 BB20-1.5s 10.54%와 거의 같은 공격형 수준이다. BB20-1.0s의 upper breach 18.92%보다는 낮으므로 “너무 좁은 밴드”로 단정할 수준은 아니다.

## 8. Test Split 확인

| 후보 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.928985 | 0.017134 | 0.053882 | 0.267381 | 10.8038 | 0.012007 | 0.000704 | -0.213559 |
| q30-b2 | 0.879588 | 0.031262 | 0.089150 | 0.228890 | 9.2485 | 0.012289 | 0.000711 | -0.210900 |
| q35-b2 | 0.826952 | 0.048838 | 0.124209 | 0.199180 | 8.0480 | 0.012297 | 0.001061 | -0.107917 |

| 기준선 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | relative_band_width | median_band_width | p90_band_width |
|---|---:|---:|---:|---:|---:|---:|---:|
| BB20-2.0s | 0.877526 | 0.064784 | 0.057690 | 0.116620 | 4.7121 | 0.096487 | 0.204873 |
| BB20-1.5s | 0.787106 | 0.110314 | 0.102580 | 0.087465 | 3.5341 | 0.072365 | 0.153655 |
| BB20-1.0s | 0.624856 | 0.193244 | 0.181900 | 0.058310 | 2.3561 | 0.048243 | 0.102437 |
| BB60-2.0s | 0.922346 | 0.043515 | 0.034139 | 0.126580 | 5.1146 | 0.110231 | 0.216355 |
| BB60-1.5s | 0.849059 | 0.081416 | 0.069524 | 0.094935 | 3.8359 | 0.082673 | 0.162266 |

test split에서는 투자 지표가 약하지만, 밴드 위치 관계는 validation과 거의 같다. q30-b2는 BB20-2.0s와 비슷한 coverage, q35-b2는 BB20-1.5s와 비슷한 공격형 breach에 머문다.

## 9. 후보 분류

| 후보 | 분류 | 판단 |
|---|---|---|
| q25-b2 | 보수형 후보 | coverage는 BB60-2.0s와 비슷하지만 절대 폭이 매우 넓다. 안정성 비교용 reserve로 유지한다. |
| q30-b2 | 기본형 후보 | coverage와 upper breach가 BB20-2.0s와 BB20-1.5s 사이에 있어 기본 후보로 가장 적합하다. |
| q35-b2 | 공격형 후보 | upper breach가 BB20-1.5s와 비슷하다. 과도하게 좁다고 단정하지 말고 100티커에서 안정성 확인이 필요하다. |

## 10. AI 밴드 위치 감각

AI 밴드는 Bollinger보다 절대 폭이 넓다. q35-b2도 validation avg_band_width 0.1863으로 BB20-1.5s 0.0767의 약 2.4배다.

다만 breach 관점에서는 q30-b2와 q35-b2가 현실적인 위치로 내려왔다. q30-b2는 기본형, q35-b2는 공격형 후보로 볼 수 있다.

중요한 비대칭은 lower breach가 Bollinger보다 낮고 upper breach가 상대적으로 높다는 점이다. 현재 PatchTST 밴드는 하방을 더 보수적으로 덮고, 상방 초과에는 더 많이 열려 있다. 다음 안정성 확인에서 이 비대칭이 섹터/시장구간별로 유지되는지 봐야 한다.

## 11. 다음 100티커 안정성 후보

1. `q30-b2`: 기본 후보. coverage 0.8907, upper breach 7.86%, IC 0.0690으로 가장 균형이 좋다.
2. `q35-b2`: 공격형 후보. q30보다 폭을 더 줄이면서 투자 지표를 유지하는지 확인할 가치가 있다.

`q25-b2`는 보수형 reserve로 남기되, 다음 100티커 smoke의 1차 후보에서는 제외해도 된다. 이유는 Bollinger 대비 폭이 너무 넓어 CP22 이전의 과보수 문제를 충분히 줄였다고 보기 어렵기 때문이다.

## 12. 결론

`q35-b2`의 upper breach 11.30%는 BB20-1.5s의 10.54%와 비슷하고 BB20-1.0s의 18.92%보다 낮다. 따라서 q35-b2는 “너무 좁은 실패 후보”가 아니라 공격형 후보로 유지한다.

기본 후보는 `q30-b2`로 둔다. `q30-b2`는 coverage 0.8907로 목표권 상단에 있고, upper breach가 BB20-2.0s와 BB20-1.5s 사이에 있으며, 투자 지표도 q25 대비 큰 손상 없이 유지된다.

다음 CP는 full 473이 아니라 `q30-b2`, `q35-b2`의 100티커 안정성 확인이 맞다.

## 13. 남은 리스크

- Bollinger는 모델 예측이 아니라 과거 return 분포 기준선이므로 투자 지표는 PatchTST와 직접 비교하지 않았다.
- test split의 fee_adjusted_return은 세 후보 모두 음수다. validation에서 후보를 고르되, 100티커와 이후 walk-forward에서 재검증해야 한다.
- AI 밴드 폭은 전통 Bollinger보다 아직 넓다. 다음 단계에서는 q30/q35가 universe 확장에서도 비슷한 breach 구조를 유지하는지 확인해야 한다.
