# CP148-0 모델 라인업 실험 계획 초안

이 문서는 CP148-0-S preflight 이후 바로 실행할 수 있는 실험 축 초안이다. 실제 W&B online 실행은 사용자가 VSCode 로컬 터미널에서 진행한다.

## 공통 금지/운영

- product run 교체 금지
- inference 저장 금지
- full 473/500 장기 학습은 각 CP 승인 후 실행
- W&B online 큰 실험은 사용자가 직접 실행

## CP148-LM-1D

- 우선 모델: PatchTST, TCNQuantile
- horizon: h5
- feature_set: full_features, no_fundamentals
- selector: line_gate
- search space 초안: lr, weight_decay, dropout, patch_len/stride 또는 TCN dilation/channel

## CP149-BM-1D

- 우선 모델: CNN-LSTM, TCNQuantile, TiDE 보조
- horizon: h5
- feature_set: price_volatility_volume, price_volatility
- selector: band_gate
- search space 초안: q_low/q_high, lambda_band, seq_len, band_mode, TCN channel/dilation

## CP150-LM-1W

- 우선 모델: PatchTST, TiDE, TCNQuantile
- horizon: h4 우선, h6 보조
- seq_len: 104
- feature_set: price_volatility_volume, no_fundamentals
- selector: line_gate

## CP151-BM-1W

- 우선 모델: CNN-LSTM, TiDE, TCNQuantile
- horizon: h4
- seq_len: 104
- feature_set: price_volatility_volume
- selector: band_gate

## W&B 명령 템플릿

```powershell
cd C:\Users\user\lens
.\.venv\Scripts\Activate.ps1
$env:WANDB_MODE='online'
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model patchtst --timeframe 1D --horizon 5 --seq-len 252 --epochs 3 --batch-size 256 --device cuda --no-compile --wandb --wandb-project lens-cp148 --feature-set full_features --checkpoint-selection line_gate
```
