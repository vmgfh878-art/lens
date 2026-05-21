# CP148-0-S-TCN-AUDIT 보고서

판정: **WARN**

이번 CP는 TCNQuantile이 1D line 실험 후보로 들어갈 구조적 자격이 있는지 확인한 감사다. full training, sweep, DB write, inference 저장, W&B/Optuna 실행은 하지 않았다.

## 1. causal 구조

- Conv1D padding: `left_only`
- chomp/trim 사용: `False`
- 미래 입력 leakage: `False`
- target leakage: `False`
- causal probe 통과: `True`

## 2. dilation / residual 구조

- kernel_size: `3`
- dilations: `[1, 2, 4, 8]`
- residual blocks: `4`
- conv layers per block: `2`
- receptive field: `61` / seq_len `252` = `0.2421`
- 해석: TCN이라고 부를 수 있는 exponential dilation 구조는 갖췄지만, 1D seq_len 252 기준 RF가 약 24%라 장기 문맥 후보로는 주의가 필요하다.

## 3. 출력 계약

- input shape: `[4, 252, 36]`
- line shape: `[4, 5]`
- lower shape: `[4, 5]`
- upper shape: `[4, 5]`
- output finite: `True`
- lower<=upper 비율: `0.5000`

## 4. normalization / RevIN 경로

- internal RevIN: `False`
- expected normalization: `preprocessing/train split mean_std`
- per-sample future normalization: `False`
- normalization target leakage: `False`

## 5. loss 연결

- ForecastCompositeLoss total finite: `True`
- total loss: `1.503828`
- line loss: `0.635609`
- band loss: `0.827358`

## 6. feature contract

- FEATURE_CONTRACT_VERSION: `v3_adjusted_ohlc`
- MODEL_N_FEATURES: `36`
- model feature columns count: `36`
- atr_ratio 모델 feature 포함: `False`

## 7. 판정

- blockers: `[]`
- warnings: `['receptive field가 seq_len 252 대비 30% 미만이라 장기 의존성 후보로는 주의 필요', '내부 RevIN 없음: 기존 train/preprocessing 표준화 계약에 의존']`
- CP148 Stage 2 exploratory 후보 포함 가능: `True`

TCNQuantile은 causal leakage와 출력/loss 계약은 통과했다. 다만 receptive field가 252일 입력 대비 짧고 내부 RevIN이 없으므로, CP148 Stage 2에서는 exploratory line 후보로만 올리고 PatchTST와 동급 주력 후보로 해석하지 않는 것이 맞다.
