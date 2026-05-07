# CP14-R PatchTST Target Baseline 2종 비교

## 1. 실행 환경
- 프로젝트 루트: `C:\Users\user\lens`
- 파이썬: `C:\Users\user\lens\.venv\Scripts\python.exe`
- 디바이스: `cuda`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- 공통 조건:
  - `model=patchtst`
  - `timeframe=1D`
  - `seq_len=252`
  - `horizon=기본값`
  - `batch_size=64`
  - `compile=false`
  - `ci_aggregate=target`
  - `ci_target_fast=false`
  - `band_mode=direct`
  - `epochs=5`
  - `wandb=off`

## 2. 실행 명령
### A. raw_future_return
```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train `
  --model patchtst `
  --timeframe 1D `
  --seq-len 252 `
  --epochs 5 `
  --batch-size 64 `
  --device cuda `
  --no-wandb `
  --no-compile `
  --ci-aggregate target `
  --line-target-type raw_future_return `
  --band-target-type raw_future_return
```

### B. volatility_normalized_return
```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train `
  --model patchtst `
  --timeframe 1D `
  --seq-len 252 `
  --epochs 5 `
  --batch-size 64 `
  --device cuda `
  --no-wandb `
  --no-compile `
  --ci-aggregate target `
  --line-target-type volatility_normalized_return `
  --band-target-type volatility_normalized_return
```

## 3. A/B 결과 표
| 항목 | A: raw_future_return | B: volatility_normalized_return |
|---|---:|---:|
| 종료 여부 | 실패 | 실패 |
| exit code | `1` | `1` |
| 첫 실패 phase | `train` | `train` |
| 첫 실패 epoch | `1` | `1` |
| 첫 실패 batch | `2` | `2` |
| 첫 실패 metric | `total_loss` | `total_loss` |
| 핵심 원인 | 입력 `features`에 `NaN` 존재 | 입력 `features`에 `NaN` 존재 |
| val_total | 산출 불가 | 산출 불가 |
| val_forecast | 산출 불가 | 산출 불가 |
| line_loss | 산출 불가 | 산출 불가 |
| band_loss | 산출 불가 | 산출 불가 |
| coverage | 산출 불가 | 산출 불가 |
| avg_band_width | 산출 불가 | 산출 불가 |
| mae | 산출 불가 | 산출 불가 |
| smape | 산출 불가 | 산출 불가 |
| direction_accuracy | 산출 불가 | 산출 불가 |
| spearman_ic | 산출 불가 | 산출 불가 |
| top_k_long_spread | 산출 불가 | 산출 불가 |
| top_k_short_spread | 산출 불가 | 산출 불가 |
| long_short_spread | 산출 불가 | 산출 불가 |
| fee_adjusted_return | 산출 불가 | 산출 불가 |
| fee_adjusted_sharpe | 산출 불가 | 산출 불가 |
| fee_adjusted_turnover | 산출 불가 | 산출 불가 |
| overprediction_rate | 산출 불가 | 산출 불가 |
| mean_overprediction | 산출 불가 | 산출 불가 |

## 4. 실패 진단
- 두 실험 모두 `epoch=1`, `batch=2`에서 처음 무너졌다.
- 공통 진단 출력:
  - `tensor[features]`의 `finite_ratio = 0.8333333134651184`
  - `prediction.line`, `prediction.lower_band`, `prediction.upper_band`가 즉시 `NaN`
  - `loss_components.total_loss`, `forecast_loss`, `line_loss`, `band_loss`, `cross_loss`가 모두 `NaN`
- 즉 target 차이보다 먼저, 입력 feature tensor가 이미 오염되어 있어 비교 실험이 성립하지 않았다.

## 5. 입력 피처 NaN 집계
- `prepare_dataset_splits(timeframe='1D', seq_len=252, horizon=5)` 기준 학습 유니버스는 `473`개 ticker다.
- `MODEL_FEATURE_COLUMNS=36` 중 `NaN`이 남아 있는 컬럼은 아래 6개다.

| 피처 | 누적 NaN 개수 |
|---|---:|
| `revenue` | 15445 |
| `net_income` | 17575 |
| `equity` | 17575 |
| `eps` | 17575 |
| `roe` | 17790 |
| `debt_ratio` | 18073 |

- 즉 PatchTST 입력에 재무 6개 컬럼 `NaN`이 그대로 남아 있고, 이게 학습 배치로 들어가면서 바로 깨진다.

## 6. band guardrail 판단
- 이번 CP에서는 guardrail 자체를 비교할 수 없었다.
- 이유:
  - 두 target 모두 validation까지 도달하지 못함
  - 따라서 `coverage`, `avg_band_width`, `band_loss` 비교가 불가능함
- 결론:
  - 이번 단계에서는 `raw target 유지` 대 `volatility target 채택` 판단을 내릴 수 없다
  - 먼저 PatchTST 입력 `NaN` 경로를 막아야 한다

## 7. 투자 지표 판단
- CP10 원칙대로 투자 지표는 raw realized return 기준으로 해석해야 한다.
- 그러나 이번에는 validation/test 요약이 만들어지지 못해서
  - `spearman_ic`
  - `top-k spread`
  - `fee_adjusted_return`
  - `direction_accuracy`
  를 비교할 수 없었다.
- 따라서 이번 CP에서 투자 지표 기반 채택 판단도 유보한다.

## 8. raw target 유지 vs volatility target 채택 여부
- 현재 결론: **판단 보류**
- 이유:
  - 두 실험 모두 같은 입력 오염으로 실패
  - target type 차이로 인한 성능/밴드 품질 비교가 아직 불가능

## 9. 추가 점검 결과
- PatchTST CUDA run exit code `0` 여부:
  - 충족하지 못함
  - 실제 학습 명령 종료코드는 두 케이스 모두 `1`
- NaN/Inf:
  - 있음
  - 입력 `features`에서 먼저 발생
- `failed_nan` 저장 정책 영향:
  - 이번 실행은 `--save-run` 없이 실행되어 DB 저장 경로에는 영향 없음
- `volatility_normalized_return` caveat:
  - 이 target은 가격 decode/시그널 생성 대상이 아니다
  - 비교가 가능해지더라도 score 용도로만 다뤄야 한다
- `patch_len/stride`:
  - 모델 init에는 있으나 CLI 미노출 상태 유지
  - 이번 CP에서는 실험하지 않음

## 10. 다음 CP 추천
- 추천 CP: **PatchTST 입력 정합 복구**
- 우선순위:
  1. `ai.preprocessing`에서 PatchTST 학습 경로에 재무 6개 `NaN`이 왜 남는지 고정
  2. PatchTST용으로 `NaN -> 0 + has_fundamentals flag`가 실제 학습 텐서까지 일관되게 들어가는지 확인
  3. 그 다음에 이번 CP14-R을 같은 명령으로 재실행

## 11. 남은 리스크
- 현재 상태로는 PatchTST solo track의 첫 비교 실험이 재현되지 않는다.
- 즉 지금은 target choice보다 먼저 데이터 정합 문제가 막고 있다.
- 이 문제를 안 잡고 다음 sweep으로 넘어가면 target 비교, patch geometry 비교, seq_len 비교가 전부 의미를 잃는다.
