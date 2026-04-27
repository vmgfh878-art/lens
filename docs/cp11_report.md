# CP11 보고

## 1. 변경 파일
- `C:\Users\user\lens\ai\models\common.py`
- `C:\Users\user\lens\ai\models\cnn_lstm.py`
- `C:\Users\user\lens\ai\loss.py`
- `C:\Users\user\lens\ai\train.py`
- `C:\Users\user\lens\ai\evaluation.py`
- `C:\Users\user\lens\ai\tests\test_architecture_postprocess.py`
- `C:\Users\user\lens\ai\tests\test_losses.py`
- `C:\Users\user\lens\ai\tests\test_cp9_5.py`

## 2. 구조 변경
- `ForecastOutput`: `direction_logit: torch.Tensor | None = None` 필드를 추가했다. 기본값은 `None`이라 기존 모델 호환은 유지된다.
- `CNN-LSTM direction_head`: `use_direction_head=True`일 때만 최종 pooled 표현에서 `[batch, horizon]` 크기의 `direction_logit`을 만든다.
- 다른 모델 영향: `PatchTST`, `TiDE`는 구조 변경 없이 기존 출력 계약만 그대로 따른다.

## 3. loss 변경
- `forecast_loss`: 기존과 동일하게 `lambda_line * line_loss + lambda_band * band_loss + lambda_cross * cross_loss`를 유지했다.
- `direction_loss`: `raw_future_returns > 0`에서 내부적으로 방향 라벨을 만들고, `BCEWithLogits` 기반 비대칭 가중치(`하락=2.0`, `상승=1.0`)를 적용했다.
- `total_loss`: `forecast_loss + lambda_direction * direction_loss`로 구성했다.
- `lambda_direction`: 기본값 `0.1`을 사용했다.

## 4. 보수적 예측 guardrail
- 5티커 1 epoch 검증 기준
  - baseline: `coverage 0.6741`, `avg_band_width 0.1506`, `band_loss 0.0339`, `overprediction_rate 0.5689`, `mean_overprediction 0.0606`
  - direction head: `coverage 0.7543`, `avg_band_width 0.1749`, `band_loss 0.0279`, `overprediction_rate 0.4014`, `mean_overprediction 0.0585`
- 50티커 짧은 검증 기준
  - baseline: `coverage 0.9053`, `avg_band_width 0.1167`, `band_loss 0.0142`, `overprediction_rate 0.3563`, `mean_overprediction 0.0274`
  - direction head: `coverage 0.9152`, `avg_band_width 0.1209`, `band_loss 0.0140`, `overprediction_rate 0.3791`, `mean_overprediction 0.0317`
- 해석: 50티커 검증에서는 `coverage`, `band_loss`는 소폭 개선됐지만 `avg_band_width`, `overprediction_rate`, `mean_overprediction`은 소폭 악화됐다. 따라서 방향성 개선을 바로 성공으로 보기보다는 추가 짧은 반복 검증이 필요하다.

## 5. 비교 실험
- baseline: `CNN-LSTM line + band`, `use_direction_head=False`
- direction_head: `CNN-LSTM line + band + direction_logit`, `use_direction_head=True`
- 동일 조건: `seed=42`, `timeframe=1D`, `line_target_type=raw_future_return`, `band_target_type=raw_future_return`, 같은 split, 같은 epoch 수, 같은 ticker subset, 같은 `batch_size=64`
- 1차 smoke: `limit_tickers=5`, `epochs=1`
- 2차 짧은 검증: `limit_tickers=50`, `epochs=1`

## 6. 투자 지표
- 5티커 1 epoch 검증
  - baseline: `direction_accuracy 0.4800`, `spearman_ic -0.0812`, `long_short_spread -0.0043`, `fee_adjusted_return -0.9495`
  - direction head: `direction_accuracy 0.5172`, `spearman_ic 0.1148`, `long_short_spread 0.0033`, `fee_adjusted_return 3.5336`
- 50티커 짧은 검증
  - baseline: `direction_accuracy 0.5298`, `spearman_ic -0.0456`, `long_short_spread -0.0041`, `fee_adjusted_return -0.9846`
  - direction head: `direction_accuracy 0.4549`, `spearman_ic -0.0345`, `long_short_spread -0.0056`, `fee_adjusted_return -0.9963`
- 해석: 5티커 smoke에서는 direction head가 좋아 보였지만, 50티커에서는 `direction_accuracy`와 `long_short_spread`, `fee_adjusted_return`이 오히려 나빠졌다. 이번 CP 목적대로 비교 판은 만들었고, 방향 head 효과는 아직 확정적으로 좋다고 말할 수 없다.

## 7. 검증 명령
- 테스트
  - `C:\Users\user\lens\.venv\Scripts\python.exe -m unittest discover -s ai\tests -p "test_*.py"`
  - `PYTHONPATH="C:\Users\user\lens\backend;C:\Users\user\lens" C:\Users\user\lens\.venv\Scripts\python.exe -m unittest backend.tests.test_feature_svc backend.tests.test_collector_jobs backend.tests.test_api`
- 5티커 smoke
  - `C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 5 --seed 42 --device cpu --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return`
  - `C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 5 --seed 42 --device cpu --no-wandb --no-compile --use-direction-head --line-target-type raw_future_return --band-target-type raw_future_return`
- 50티커 짧은 검증
  - `C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 50 --seed 42 --device cpu --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return`
  - `C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 50 --seed 42 --device cpu --no-wandb --no-compile --use-direction-head --line-target-type raw_future_return --band-target-type raw_future_return`

## 8. 남은 리스크
- `CNN-LSTM + CUDA + bf16 autocast` 경로에서는 검증 지표가 `NaN`으로 무너지는 문제가 있다. 이번 비교 실험은 같은 조건의 CPU 경로로 고정해서 수행했다.
- 현재 early stopping 최선 기준은 `forecast_loss`로 분리해 두었지만, direction head의 실전 효용은 1 epoch 짧은 검증만으로 확정할 수 없다.
- 50티커 검증에서 `coverage`는 좋아졌지만 `avg_band_width`, `overprediction_rate`, `mean_overprediction`, `fee_adjusted_return`은 개선이 약하거나 악화됐다. 다음 단계에서는 2~3 epoch 짧은 반복과 GPU `NaN` 원인 분리가 필요하다.
