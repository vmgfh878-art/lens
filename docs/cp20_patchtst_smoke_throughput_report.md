# CP20-R PatchTST target smoke + batch 처리량 측정

## 1. 실행 환경

- 작업일: 2026-04-28
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- Python: `C:\Users\user\lens\.venv\Scripts\python.exe`
- 공통 조건: `model=patchtst`, `timeframe=1D`, `seq_len=252`, `device=cuda`, `--no-wandb`, `--no-compile`, `ci_aggregate=target`
- 금지 조건 확인: 전체 473티커 5epoch 실행 안 함, 구조 변경 안 함, `--save-run` 안 씀

## 2. 1단계 volatility target finite smoke

명령: `volatility_normalized_return`, `limit_tickers=50`, `batch_size=64`, `epochs=1`

| 항목 | 결과 |
|---|---:|
| 판정 | PASS |
| 종료코드 | 0 |
| epoch_seconds | 100.796 |
| 총 소요 시간 | 203.95초 |
| VRAM peak | 5063 MB |
| VRAM delta | 1896 MB |
| val_total | 1.725217 |
| coverage | 0.812316 |
| avg_band_width | 4.093604 |
| spearman_ic | -0.001916 |
| fee_adjusted_return | -0.959846 |

해석: volatility target 경로는 NaN 없이 정상 동작한다. 다만 이 target은 가격 decode와 signal 저장 대상이 아니므로, 투자 지표는 raw realized return 기준 해석만 가능하다.

## 3. 2단계 raw target batch ladder

각 실행은 `raw_future_return`, `limit_tickers=50`, `epochs=1` 조건이다.

| batch_size | 판정 | 종료코드 | epoch_seconds | 총 소요 시간 | VRAM peak | VRAM delta | val_total | coverage | avg_band_width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | PASS | 0 | 94.516 | 167.30초 | 5143 MB | 2004 MB | 0.113210 | 1.000000 | 1.052239 |
| 128 | PASS | 0 | 89.544 | 161.15초 | 6724 MB | 3549 MB | 0.168098 | 1.000000 | 1.515171 |
| 256 | PASS | 0 | 85.873 | 171.31초 | 9977 MB | 6849 MB | 0.246432 | 0.997386 | 2.048759 |

판단: `batch_size=256`은 OOM/NaN 없이 통과했고 epoch 시간도 가장 짧다. 다만 64 대비 epoch 시간 개선은 약 9.1%라 크지는 않다. 16GB 안에서는 안정적으로 들어가므로 다음 짧은 비교의 운영 후보는 `256`으로 둔다.

## 4. 3단계 raw vs volatility 3epoch 비교

공통 조건: `limit_tickers=50`, `batch_size=256`, `epochs=3`

| target | 판정 | 종료코드 | 총 소요 시간 | VRAM peak | epoch_seconds |
|---|---|---:|---:|---:|---|
| raw_future_return | PASS | 0 | 322.37초 | 10086 MB | 84.438, 83.097, 84.406 |
| volatility_normalized_return | PASS | 0 | 336.50초 | 9820 MB | 90.106, 88.891, 88.850 |

## 5. 3epoch 비교 지표

아래 표는 validation best metrics 기준이다.

| metric | raw_future_return | volatility_normalized_return |
|---|---:|---:|
| coverage | 1.000000 | 0.808691 |
| avg_band_width | 0.847906 | 4.032927 |
| band_loss | 0.084538 | 0.619130 |
| mae | 0.077523 | 1.242704 |
| smape | 1.620633 | 1.577995 |
| spearman_ic | 0.075956 | -0.005644 |
| long_short_spread | 0.005186 | -0.000746 |
| fee_adjusted_return | 6.354306 | -0.876294 |
| direction_accuracy | 0.455211 | 0.438596 |

해석: volatility target은 finite smoke와 3epoch 모두 통과했지만, 밴드 폭이 raw 대비 크게 넓고 `band_loss`, `spearman_ic`, `long_short_spread`, `fee_adjusted_return`이 모두 불리하다. 현재 조건에서는 raw target 유지가 맞다.

## 6. 총 소요 시간

- 성공한 측정 실행 합계: 약 1362.58초, 약 22.7분
- 로그 파일:
  - `docs/cp20_vol_smoke_b64.stdout.log`
  - `docs/cp20_raw_b64.stdout.log`
  - `docs/cp20_raw_b128.stdout.log`
  - `docs/cp20_raw_b256.stdout.log`
  - `docs/cp20_raw_3epoch_b256.stdout.log`
  - `docs/cp20_vol_3epoch_b256.stdout.log`

## 7. 다음 full run 가능 여부

가능하다. 단, 이번 smoke 기준에서는 다음 full run target은 `raw_future_return`으로 제한하는 편이 낫다. `volatility_normalized_return`은 실행 가능성은 확인됐지만, 50티커 3epoch에서 밴드 품질과 투자 지표가 모두 약하다.

## 8. full run 예상 시간

50티커 제한 실행에서 실제 eligible은 48개였고, 전체 1D 학습 대상은 473개다. 단순 스케일링 배율은 약 9.85배다.

- `batch_size=256`, raw target 기준 예상 epoch 시간: 약 13.8분
- `5 epoch` 예상 순수 epoch 시간: 약 69분
- 데이터 준비와 validation/test 평가 포함 예상 총 시간: 약 80분에서 90분

## 9. 결론

- volatility target 50티커 1epoch: PASS
- raw target batch ladder: `64/128/256` 모두 PASS
- 안정 batch 후보: `256`
- 50티커 3epoch 비교: 가능
- 다음 full run 판단: `raw_future_return`, `batch_size=256`으로 진행 가능. 다만 성능 목적의 긴 실험 전에는 `patch_len/stride` 노출 여부를 먼저 정리하는 편이 더 효율적이다.
