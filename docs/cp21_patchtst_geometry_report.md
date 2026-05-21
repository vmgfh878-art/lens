# CP21-R PatchTST geometry smoke 보고

## 1. 실행 환경

- 날짜: 2026-04-28
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- Python: `C:\Users\user\lens\.venv\Scripts\python.exe`
- 공통 조건: `--model patchtst --timeframe 1D --seq-len 252 --epochs 3 --batch-size 256 --device cuda --no-wandb --no-compile --ci-aggregate target --line-target-type raw_future_return --band-target-type raw_future_return --limit-tickers 50`
- 금지 조건 확인: 전체 473티커 5epoch 미실행, direction/rank head 미추가, TiDE/CNN-LSTM 미수정, `--save-run` 미사용

## 2. 구현 변경

- `ai/train.py`에 PatchTST 전용 CLI 인자를 추가했다.
- 추가 인자: `--patch-len`, `--patch-stride`, `--patchtst-d-model`, `--patchtst-n-heads`, `--patchtst-n-layers`
- 기본값은 현재 PatchTST 구현과 동일하다: `patch_len=16`, `stride=8`, `d_model=128`, `n_heads=8`, `n_layers=3`
- 해당 인자는 `model=patchtst`일 때만 모델 생성 경로에 전달된다.

## 3. 로그 변경

- epoch 로그에 `elapsed_seconds`, `estimated_remaining_seconds`, `vram_peak_allocated_mb`를 추가했다.
- stdout 한 줄 요약에도 `epoch_seconds`, `elapsed_seconds`, `eta_seconds`, `vram_peak_allocated_mb`가 보이도록 했다.
- W&B는 이번 smoke에서 사용하지 않았다.

## 4. geometry 결과표

| 조건 | patch_len | stride | n_patches | 평균 epoch_seconds | VRAM peak | coverage | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return | direction_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 16 | 8 | 30 | 75.9685 | 7781 MB | 0.999954 | 0.661335 | 0.065957 | 0.069970 | 0.005070 | 6.002602 | 0.458476 |
| B short | 8 | 4 | 62 | 136.2211 | 13464 MB | 1.000000 | 1.631322 | 0.162684 | 0.047372 | 0.002970 | 0.488734 | 0.437235 |
| C dense | 16 | 4 | 60 | 132.1239 | 13039 MB | 1.000000 | 1.838474 | 0.183312 | -0.015071 | -0.003404 | -0.972538 | 0.438860 |
| D long | 32 | 16 | 14 | 45.6483 | 4976 MB | 0.988010 | 0.352508 | 0.035526 | 0.030214 | 0.003971 | 2.140137 | 0.463757 |
| E overlap | 32 | 8 | 28 | 71.5938 | 7472 MB | 1.000000 | 0.943122 | 0.094038 | 0.062391 | 0.004038 | 2.314449 | 0.444315 |

## 5. band guardrail 판단

- 모든 조건이 NaN/OOM 없이 통과했다.
- 모든 조건의 coverage가 0.98 이상이라 경고 구간이다. 특히 `coverage=1.0`은 좋은 결과로 단정하지 않는다.
- D long은 밴드 폭과 band_loss가 가장 좋고 속도도 가장 빠르지만, A baseline보다 IC와 long-short spread가 떨어진다.
- A baseline은 밴드가 과하게 보수적인 편이지만 투자 지표는 가장 좋다.

## 6. 투자 지표 판단

- `spearman_ic`, `long_short_spread`, `fee_adjusted_return` 기준으로는 A baseline이 가장 낫다.
- D long은 속도와 밴드 폭 관점의 후보지만, 투자 지표 개선 후보는 아니다.
- B short와 C dense는 patch 수가 많아져 느리고 VRAM 사용량이 커졌으며 지표도 개선되지 않았다.
- E overlap은 baseline보다 빠르고 IC가 크게 무너지진 않았지만, baseline을 이기지는 못했다.

## 7. 채택 여부

현재 smoke 기준으로는 geometry 변경을 채택하지 않는다. 다음 기준 설정은 `patch_len=16`, `stride=8` baseline을 유지한다.

보조 후보는 `patch_len=32`, `stride=16`이다. 이 조건은 full run용 주력 후보가 아니라, 밴드 폭과 속도 개선이 필요한 경우의 calibration 후보로 둔다.

## 8. 다음 CP 추천

- full run으로 바로 가지 말고, baseline geometry에서 coverage가 1.0에 붙는 문제를 먼저 다룬다.
- 다음 후보 작업은 band calibration 쪽이다. 예: `lambda_band`, `lambda_cross`, q_low/q_high 또는 postprocess guardrail 점검.
- Patch geometry만으로는 밴드 과보수와 투자 지표를 동시에 개선하지 못했다.

## 9. 로그 파일

- `docs/cp21_A_baseline_p16_s8.stdout.log`
- `docs/cp21_B_short_p8_s4.stdout.log`
- `docs/cp21_C_dense_p16_s4.stdout.log`
- `docs/cp21_D_long_p32_s16.stdout.log`
- `docs/cp21_E_overlap_p32_s8.stdout.log`
