# CP16-R PatchTST 개선 리서치 보고서

## 1. 결론 요약

이번 리서치의 결론은 단순하다. Phase 1에서 모델을 늘릴 이유는 아직 없다. 지금 Lens에 필요한 것은 PatchTST를 더 복잡하게 만드는 것이 아니라, 현재 출력 목표인 보수적 line과 AI band가 실제 target 스케일, RevIN 복원, channel aggregation, calibration 평가와 일관되게 맞물리는지 검증하는 것이다.

### 바로 CP로 만들 후보

1. **RevIN denorm과 target 스케일 재검토**
   - 현재 PatchTST는 입력 전체를 RevIN으로 정규화하고, 출력은 `target_channel_idx=0` 기준으로 복원한다. 그런데 0번 입력 채널은 `log_return`이고, 학습 target은 기본적으로 anchor close 대비 `raw_future_return`이다. PatchTST 원논문은 같은 시계열 채널을 미래로 예측한 뒤 같은 채널 통계로 복원한다. Lens는 입력 채널과 출력 target 정의가 다르므로 이 지점이 가장 의심스럽다.
   - 바로 실험할 값: 현행, RevIN 입력 norm만 사용하고 출력 denorm 끔, target을 history volatility scale로 맞춘 뒤 후처리에서 raw return 복원.

2. **AI band validation calibration**
   - q10/q90 pinball loss만으로 실사용 coverage가 목표 coverage와 맞는다는 보장은 없다. validation residual로 horizon별, regime별 보정폭을 추정하는 conformalized quantile regression 방식이 Lens band 정체성에 가장 직접적으로 맞는다.
   - 바로 실험할 값: 전체 validation 보정, horizon별 보정, `regime_calm/neutral/stress`별 보정.

3. **보수적 line 안정화**
   - 현재 line은 `AsymmetricHuberLoss(beta=2)`로 과대 예측을 더 벌한다. 이 방향은 Lens 정체성과 맞다. 다만 RevIN 복원과 target mismatch가 있으면 asymmetric loss 효과가 흐려질 수 있다.
   - 바로 실험할 값: validation 기반 conservative offset, beta 2/3/4, horizon별 offset. direction accuracy 단독 개선은 채택 기준이 아니다.

4. **Patch geometry와 lookback sweep**
   - 공식 구현의 대표 supervised 설정은 `patch_len=16`, `stride=8`, `seq_len=336` 또는 `512` 계열이다. Lens는 `seq_len=252`, `patch_len=16`, `stride=8`에 가깝지만, 현재 train CLI에서 `patch_len/stride/d_model/n_layers/n_heads`가 직접 열려 있지 않다.
   - 바로 실험할 값: `patch_len/stride = 8/4, 16/8, 24/12`, 그 뒤 `seq_len = 126, 252, 504`.

5. **channel aggregation 검증**
   - 공식 PatchTST는 채널별 예측을 만들고 채널별 loss를 평균한다. Lens는 채널별 hidden을 하나로 집계해 단일 target을 예측한다. `ci_aggregate=target`은 target 채널 hidden만 쓰므로 macro/fundamental 등 다른 채널을 인코딩해도 최종 예측에는 거의 반영되지 않을 수 있다.
   - 바로 실험할 값: `target`, `mean`, `attention`. `ci_target_fast=True`는 속도 비교용으로만 쓰고 fidelity 기본값으로 올리면 안 된다.

### 보류할 후보

- **self-supervised pretraining**: 논문과 공식 구현에서 효과가 있었지만, Phase 1에서는 supervised target, RevIN, band calibration 문제가 먼저다. pretraining은 target과 calibration이 안정된 뒤 Phase 1.5 또는 별도 연구 CP로 보류한다.
- **market excess return**: 금융적으로 매력적이지만 현재 `market_excess_return`은 benchmark future return 경로가 아직 연결되지 않았다. SPY 또는 시장 proxy 정합성 CP 이후가 맞다.
- **rank target과 ranker**: ranking/IC 평가는 중요하지만 Phase 1의 주 모델 출력은 보수적 line과 band다. ranker는 PatchTST 교체나 모델 추가가 아니라 보조 평가 또는 Phase 1.5 후보로만 둔다.

### 하지 말아야 할 후보

- TiDE/CNN-LSTM 재개 또는 새 모델 추가.
- direction accuracy만 보고 채택.
- `ci_target_fast=True`를 성능 검증 없이 기본값으로 승격.
- band calibration 없이 q10/q90 pinball loss만 믿고 제품 지표로 사용.
- “나중에 앙상블” 같은 결론. Phase 1은 PatchTST 단일 축이다.

## 2. Lens 현재 상태와 문제 정의

### 현재 목표

Lens Phase 1의 주 출력은 다음 세 가지다.

- 하방 보수적 예측선: 과대 예측을 더 위험하게 보는 line.
- AI 기반 band 리스크 지표: q_low/q_high 구간과 coverage, width.
- direction/rank/backtest: 모델 채택을 돕는 보조 지표.

따라서 모델 개선의 우선순위는 `direction_accuracy`가 아니라 `coverage`, `avg_band_width`, `band_loss`, `overprediction_rate`, `mean_overprediction`, `spearman_ic`, `top-k spread`, `fee-adjusted` 지표의 묶음이어야 한다.

### 현재 지표

현재 `ai/evaluation.py`는 다음 지표를 계산한다.

- band: `coverage`, `avg_band_width`
- error: `mae`, `smape`
- 보수성: raw target일 때 `mean_signed_error`, `overprediction_rate`, `mean_overprediction`
- rank/backtest 보조: `spearman_ic`, `top_k_long_spread`, `top_k_short_spread`, `long_short_spread`, `fee_adjusted_return`, `fee_adjusted_sharpe`, `fee_adjusted_turnover`

이 방향은 좋다. 다만 band가 “모델이 낸 q10/q90”인지 “calibrated interval”인지가 아직 분리되어 있지 않다. 실사용 band는 후자가 되어야 한다.

### 현재 병목 추정

가장 의심되는 병목은 세 가지다.

1. **RevIN denorm과 target 정의 mismatch**
   - `ai/models/patchtst.py`는 output을 `target_channel_idx` 통계로 denormalize한다.
   - `backend/app/services/feature_svc.py` 기준 0번 feature는 `log_return`이다.
   - `ai/preprocessing.py`의 기본 target은 anchor close 대비 future return이다.
   - 즉 모델은 log-return 입력 통계로 cumulative simple return target을 복원하고 있을 가능성이 있다. 이 추정이 틀렸다면 틀렸다고 확인해야 한다. 하지만 코드 흐름상 가장 먼저 검증할 가치가 있다.

2. **channel independence는 있지만, 공식 구현식 채널별 출력 평균은 아니다**
   - 공식 PatchTST는 각 채널이 독립 forward를 거쳐 각 채널 미래값을 예측하고 loss를 평균한다.
   - Lens는 채널별 hidden을 단일 hidden으로 집계해 line/band head에 넣는다.
   - `ci_aggregate=target`이면 target 채널 hidden만 쓰므로 다른 feature 채널의 정보가 버려질 수 있다. 반대로 `mean/attention`은 다른 채널을 살릴 수 있지만 잡음도 함께 섞는다.

3. **band calibration이 학습 손실에만 맡겨져 있다**
   - q10/q90 pinball loss는 필요한 기본기지만, 금융 시계열의 regime shift와 fat tail에서는 empirical coverage가 목표 coverage에서 벗어날 수 있다.
   - Lens의 AI band는 제품 정체성 자체이므로 validation 기반 보정이 필요하다.

### 입력 NaN 이슈와 별개로 봐야 할 모델링 이슈

CP14-R에서는 feature NaN이 학습을 막았고, 현재 코드에는 fundamental imputation과 finite contract가 들어와 있다. 그러나 입력 NaN 해결은 “학습 가능 상태”를 만드는 문제이고, 아래 모델링 이슈는 별개다.

- target과 RevIN 복원 스케일이 맞는가.
- band가 실제 목표 coverage를 맞추는가.
- `ci_aggregate=target`이 multivariate feature를 사실상 버리는가.
- `raw_future_return`이 Lens 목적에 더 맞는가, 아니면 volatility-normalized target이 더 안정적인가.
- patch geometry와 lookback이 금융 horizon에 맞는가.

## 3. PatchTST 논문/공식 구현에서 배울 점

### 적용 가능

1. **patch_len=16, stride=8은 강한 기준선이다**
   - 원논문은 PatchTST/42와 PatchTST/64를 제안하며 둘 다 `patch length P=16`, `stride S=8`을 사용한다. 공식 supervised script도 ETTh1, ETTm1, Weather에서 주로 `patch_len=16`, `stride=8`을 쓴다.
   - Lens 기본값도 이 조합이므로 “기본값이 틀렸다”가 아니라 “이 값 주변만 좁게 열자”가 맞다.

2. **긴 lookback을 실험해야 한다**
   - 논문은 patching 덕분에 attention token 수를 줄이고 더 긴 history를 볼 수 있다고 설명한다. 논문 결과에서도 lookback을 늘렸을 때 성능이 좋아지는 사례를 제시한다.
   - Lens 1D는 `252`가 자연스러운 1년 기준이다. 다음 후보는 `126`, `252`, `504`다. `504`는 비용이 크므로 patch geometry와 batch 처리량이 확인된 뒤에 열어야 한다.

3. **channel independence는 유지하되 aggregation은 Lens식으로 검증해야 한다**
   - 논문은 channel independence가 channel mixing보다 과적합이 적고 학습 데이터가 적을 때 유리할 수 있다고 분석한다.
   - Lens는 종목별 feature가 서로 다른 스케일과 의미를 갖는다. channel independence는 맞지만, 단일 target 예측에서는 어떤 hidden을 쓰는지가 핵심이다.

4. **RevIN은 입력 비정상성 방어 장치로 유지하되 output denorm은 재검토해야 한다**
   - 논문은 각 time series instance를 평균 0, 표준편차 1로 정규화하고 output prediction에 평균과 표준편차를 다시 더한다고 설명한다.
   - Lens target이 입력 target channel과 동일한 미래값이 아니면, 그대로 복원하는 것이 오히려 손실을 왜곡할 수 있다.

5. **supervised setting부터 안정화**
   - 공식 구현의 supervised learning은 MSE/MAE benchmark 중심이다. Lens는 MSE가 아니라 line/band 목적이 다르므로 head와 loss는 Lens 정체성을 유지해야 한다.

### 조건부 가능

1. **self-supervised pretraining**
   - 논문은 masked patch reconstruction 기반 self-supervised pretraining과 fine-tuning/linear probing 성능을 제시한다.
   - 하지만 Lens는 label이 부족한 문제가 아니라 target 정의와 calibration 문제가 먼저다. pretraining을 붙이면 무엇이 좋아졌는지 해석하기 어렵다.

2. **BatchNorm vs LayerNorm**
   - 공식 backbone은 BatchNorm 계열 옵션을 사용한다. Lens는 PyTorch `TransformerEncoderLayer`와 LayerNorm 기반이다.
   - 이 차이는 논문 fidelity 후보지만, RevIN/target/band calibration보다 우선순위가 낮다.

3. **decomposition**
   - 공식 구현에는 decomposition 옵션이 있다. 하지만 모델 구조 확장으로 해석될 수 있고, DLinear/NHITS 아이디어와 겹친다. Phase 1에서는 patch/target/calibration 이후다.

### 지금은 보류

- masked pretraining pipeline 구축.
- PatchTST backbone 대규모 교체.
- iTransformer식 variate-token attention 도입. 아이디어는 좋지만 Phase 1에서는 모델 구조 변경 폭이 크다.
- DLinear/NLinear/NHITS를 새 모델로 추가.

## 4. 금융 시계열/실무 사례에서 배울 점

### target

- **raw future return**
  - 장점: 제품의 가격 overlay, 보수적 line, raw backtest와 직접 연결된다.
  - 단점: 종목별 변동성 차이와 regime shift 때문에 loss가 고변동 종목에 끌릴 수 있다.
  - 판단: Phase 1의 기본 target으로 유지해야 한다.

- **volatility-normalized return**
  - 장점: 종목 간 스케일 차이를 줄이고 cross-section rank 안정성을 높일 수 있다.
  - 단점: 가격 decode와 band overlay에 바로 쓰기 어렵다. raw return 지표와 분리해서 해석해야 한다.
  - 판단: “score target”으로 비교할 가치는 높다. 다만 제품 표시용 line/band는 raw return으로 복원 또는 별도 calibrated band가 필요하다.

- **market excess return**
  - 장점: 시장 beta가 강한 구간에서 stock-specific signal을 분리할 수 있다.
  - 단점: 현재 구현은 benchmark future return 경로가 연결되지 않았다.
  - 판단: 지금은 보류. SPY 또는 market proxy 결합 CP 이후.

### loss

- line은 현재 asymmetric Huber 방향이 맞다. 과대 예측을 더 위험하게 보는 철학은 Lens 정체성과 일치한다.
- band는 pinball loss가 기본이다. 다만 q10/q90이 empirical q10/q90을 맞추는지 별도 calibration이 필요하다.
- `lambda_width`는 현재 CLI 호환 인자이지만 실제 composite loss 계산에는 사용되지 않는다. band 폭 제어가 필요하면 loss보다 validation calibration과 interval score 평가를 먼저 넣는 것이 해석성이 좋다.

### calibration

- conformalized quantile regression은 quantile regression과 conformal prediction을 결합해 finite-sample coverage 보장을 노리는 방법이다.
- 금융 시계열은 exchangeability 가정이 약하다. 따라서 단순 split conformal을 “보장”이라고 말하면 안 된다. 대신 temporal validation, rolling calibration, adaptive conformal을 써서 coverage drift를 추적해야 한다.
- Lens에 바로 맞는 방식은 다음이다.
  - validation set에서 horizon별 score를 계산한다.
  - 하방 score: `lower_pred - actual`이 너무 높아 actual이 lower 아래로 빠진 정도를 보정한다.
  - 상방 score: `actual - upper_pred`가 너무 높아 actual이 upper 위로 나간 정도를 보정한다.
  - 목표 coverage에 맞는 quantile offset을 산출한다.
  - global offset과 regime별 offset을 모두 기록하고 test에서 비교한다.

### backtest metric

- 금융 모델은 단순 RMSE/MAE보다 rank와 비용 반영 성과가 중요하다.
- Lens가 이미 `spearman_ic`, `top-k spread`, `long_short_spread`, `fee_adjusted_return`, `fee_adjusted_sharpe`, `turnover`를 갖고 있는 것은 좋은 방향이다.
- 단, 모델 채택 조건은 다음처럼 묶어야 한다.
  - band coverage가 목표 범위를 심하게 벗어나지 않는다.
  - avg band width가 무작정 넓어지지 않는다.
  - overprediction이 줄거나 유지된다.
  - IC/top-k/fee-adjusted 중 최소 하나 이상이 개선된다.

### feature handling

- financial feature는 결측과 regime shift가 많다. 현재 fundamental NaN imputation은 필요하다.
- `has_fundamentals`, `has_macro`, `has_breadth`, regime flag는 TFT의 variable selection 아이디어처럼 “언제 어떤 feature를 믿을지”에 쓰일 수 있다.
- Phase 1에서 TFT를 구현하지 말고, attention aggregation의 weight 기록 또는 regime별 calibration으로만 아이디어를 빌리는 것이 맞다.

## 5. 변경 후보 표

| 우선순위 | 변경 후보 | 기대 효과 | 위험 | 구현 난이도 | 필요한 실험 |
|---|---|---|---|---|---|
| 1 | RevIN output denorm ablation | target scale mismatch 제거, NaN 이후 첫 성능 병목 확인 | denorm을 끄면 PatchTST 논문 fidelity 일부 감소 | 낮음 | 현행 vs 입력 RevIN만 vs target-scaled return, raw 지표와 coverage 비교 |
| 2 | Validation 기반 band calibration | coverage 목표 정렬, AI band 신뢰도 개선 | 보정폭이 과도하면 band가 넓어짐 | 중간 | global, horizon별, regime별 conformal offset 비교 |
| 3 | 보수적 line posthoc offset과 beta sweep | overprediction_rate와 mean_overprediction 안정화 | 너무 보수적이면 IC/top-k spread 하락 | 낮음 | beta 2/3/4, validation quantile offset, fee-adjusted 지표 동시 비교 |
| 4 | Patch geometry sweep | PatchTST 핵심 축 검증, 장기 history 효율 개선 | 조합이 많아지면 실험 해석이 흐려짐 | 중간 | `8/4`, `16/8`, `24/12`를 같은 target과 seed로 비교 |
| 5 | `seq_len` 126/252/504 비교 | 금융 lookback 민감도 확인 | 504는 느리고 오래된 regime noise 유입 가능 | 중간 | best patch 고정 후 126/252/504, epoch time과 VRAM 기록 |
| 6 | `ci_aggregate` target/mean/attention 비교 | multivariate feature 사용 여부 검증 | mean/attention이 잡음을 섞을 수 있음 | 낮음 | target_fast 끔, target/mean/attention 비교, attention weight 저장은 선택 |
| 7 | batch size 64/128/256 처리량 ladder | 같은 실험을 빠르게 반복할 수 있음 | 너무 큰 batch는 일반화 저하 가능 | 낮음 | VRAM, epoch time, val_pinball, coverage 비교 |
| 8 | volatility-normalized target 재실험 | cross-section rank와 일반화 개선 가능 | raw overlay decode와 해석이 어려움 | 낮음 | raw target과 score-only target 분리, raw realized return 기준 평가 |
| 9 | market excess return 연결 | 시장 공통 움직임 제거 | benchmark 누수와 정합성 위험 | 중간 | SPY/market proxy 결합 후에만 실험 |
| 10 | self-supervised pretraining | large dataset에서 fine-tuning 성능 개선 가능 | 현재 병목을 가리고 CP 범위가 커짐 | 높음 | Phase 1 보류, 안정된 supervised baseline 이후 |

## 6. 추천 CP 순서

### CP16-R.1: RevIN/target sanity CP

목표: 입력 NaN 해결 후 바로 실행 가능한 첫 PatchTST 실험.

- `raw_future_return`, `seq_len=252`, `patch_len=16`, `stride=8`, `ci_aggregate=target`, `ci_target_fast=False`를 기준선으로 둔다.
- 다음 세 가지를 비교한다.
  - 현행 RevIN denorm.
  - 입력 RevIN norm은 유지하되 output denorm 제거.
  - volatility-normalized target을 score-only로 학습하고 raw realized 지표로 평가.
- 채택 기준:
  - `coverage`, `avg_band_width`, `overprediction_rate`, `mean_overprediction`이 악화되지 않아야 한다.
  - `spearman_ic`, `long_short_spread`, `fee_adjusted_return` 중 최소 하나가 개선되면 다음 CP 후보가 된다.

### CP16-R.2: Band calibration CP

목표: Lens 정체성인 AI band를 제품 지표로 신뢰할 수 있게 만든다.

- validation prediction으로 horizon별 lower/upper miss residual을 모은다.
- global conformal offset, horizon별 offset, regime별 offset을 비교한다.
- test에서 coverage target과 avg band width trade-off를 기록한다.
- output에는 raw band와 calibrated band를 분리해서 저장한다. 제품 화면은 calibrated band를 우선 후보로 본다.

### CP16-R.3: Conservative line CP

목표: 하방 보수적 line을 안정화한다.

- asymmetric Huber `beta`를 2/3/4로 비교한다.
- validation 기반 conservative offset을 horizon별로 계산한다.
- line이 너무 내려가서 top-k spread가 죽는지 확인한다.
- 채택 기준은 direction accuracy가 아니라 overprediction 완화와 fee-adjusted 지표의 동시 유지다.

### CP16-R.4: Patch geometry와 channel aggregation CP

목표: PatchTST 논문 fidelity 축을 좁게 연다.

- `patch_len/stride`: `8/4`, `16/8`, `24/12`.
- best geometry 고정 후 `seq_len`: `126`, `252`, `504`.
- 마지막에 `ci_aggregate`: `target`, `mean`, `attention`.
- `ci_target_fast=True`는 속도/손상 확인용 보조 실험으로만 둔다.

## 7. 하지 말아야 할 것

- **새 모델 구현 금지**: DLinear, NLinear, iTransformer, TFT, NHITS, LightGBM/CatBoost ranker는 Phase 1에서 모델 추가 대상이 아니다. 배울 점만 가져온다.
- **direction head 우선 금지**: direction accuracy는 보조 지표다. band와 보수적 line이 약해지면 채택하지 않는다.
- **self-supervised pretraining 선행 금지**: 논문상 강한 후보지만, 현재 Lens 병목은 pretraining 부족이 아니라 target/denorm/calibration 불일치일 가능성이 크다.
- **`ci_target_fast=True` 기본값 금지**: 이 옵션은 target 채널만 backbone에 넣기 때문에 multivariate PatchTST fidelity를 깬다. 빠르다는 이유로 기본값이 되면 안 된다.
- **market excess return 즉시 도입 금지**: benchmark 경로가 없으면 누수와 정합성 오류가 생긴다.
- **coverage 없이 band 채택 금지**: q10/q90 pinball loss가 낮아도 empirical coverage가 맞지 않으면 제품 band가 아니다.
- **긴 sweep 선행 금지**: LR, d_model, n_layers 대형 sweep보다 RevIN/target/patch geometry가 먼저다.

## 8. 참고 자료

### PatchTST 원논문과 공식 구현

- [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, arXiv](https://arxiv.org/abs/2211.14730)
  - 핵심 요약: PatchTST는 patching과 channel independence를 핵심으로 한다. patching은 token 수를 줄여 긴 lookback을 가능하게 하고, channel independence는 각 채널을 독립 시계열로 처리한다.
- [PatchTST 논문 PDF](https://arxiv.org/pdf/2211.14730)
  - 핵심 요약: 논문은 `P=16`, `S=8`, `L=336/512` 계열을 대표 설정으로 제시한다. RevIN류 instance normalization은 입력 채널을 정규화한 뒤 예측에 평균과 표준편차를 되돌리는 방식이다.
- [yuqinie98/PatchTST 공식 구현](https://github.com/yuqinie98/PatchTST)
  - 핵심 요약: supervised와 self-supervised 폴더가 분리되어 있고, 공식 README는 PatchTST/42, PatchTST/64, self-supervised pretraining 결과를 소개한다.
- [공식 supervised Weather script](https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/weather.sh)
  - 핵심 요약: `seq_len=336`, `patch_len=16`, `stride=8`, `d_model=128`, `e_layers=3`, `batch_size=128`, `learning_rate=0.0001` 설정을 사용한다.
- [공식 supervised backbone](https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/layers/PatchTST_backbone.py)
  - 핵심 요약: RevIN, end padding, patch projection, shared Transformer, flatten head가 구현되어 있다. 출력은 채널별 미래값을 만든다.
- [공식 self-supervised pretrain](https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_self_supervised/patchtst_pretrain.py)
  - 핵심 요약: masked patch reconstruction을 사용하며 기본 `context_points=512`, `patch_len=12`, `stride=12`, `mask_ratio=0.4`가 보인다.

### 대체 아이디어에서 빌릴 점

- [LTSF-Linear 공식 구현](https://github.com/cure-lab/LTSF-Linear)
  - 배울 점: DLinear는 trend/remainder 분해, NLinear는 last-value normalization으로 distribution shift를 단순하게 방어한다. Lens에는 새 모델이 아니라 target normalization, baseline sanity, feature scale 점검 아이디어만 가져온다.
- [iTransformer, arXiv](https://arxiv.org/abs/2310.06625)
  - 배울 점: temporal token에 모든 variate를 섞으면 variate-centric representation이 약해질 수 있다는 문제 제기. Lens에서는 `ci_aggregate`를 검증하는 근거로만 사용한다.
- [iTransformer 공식 구현](https://github.com/thuml/iTransformer)
  - 배울 점: variate token으로 multivariate correlation을 attention으로 보려는 방향. Phase 1 모델 변경 대상은 아니다.
- [Temporal Fusion Transformer, Google Research](https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/)
  - 배울 점: static/known future/historical covariate를 구분하고 feature selection과 gating을 둔다. Lens에서는 feature flag, regime calibration, aggregation 해석성에만 빌린다.
- [NHITS, Nixtla 문서](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html)
  - 배울 점: 장기 예측에서 volatility와 계산 복잡도를 multi-rate pooling과 hierarchical interpolation으로 다룬다. Lens에는 multi-scale patch 후보를 나중에 검토하는 정도로만 남긴다.

### Band calibration과 conformal prediction

- [Conformalized Quantile Regression, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
  - 핵심 요약: conformal prediction과 quantile regression을 결합해 finite-sample coverage와 heteroscedastic interval 적응성을 함께 노린다. Lens band calibration의 가장 직접적인 근거다.
- [Adaptive Conformal Inference Under Distribution Shift, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)
  - 핵심 요약: exchangeability가 약한 distribution shift 상황에서 coverage frequency를 온라인으로 맞추는 접근이다. 금융 regime drift에 맞다.
- [Conformal prediction interval for dynamic time-series, ICML 2021](https://proceedings.mlr.press/v139/xu21h.html)
  - 핵심 요약: dynamic time-series에서 exchangeability 없이도 sequential prediction interval을 구성하는 EnbPI를 제안한다. Lens에는 full EnbPI 구현보다 rolling residual calibration 아이디어가 먼저다.

### 금융 평가와 실무형 방어선

- [Stock return prediction with multiple measures using neural network models, Financial Innovation 2024](https://link.springer.com/article/10.1186/s40854-023-00608-w)
  - 핵심 요약: stock return prediction에서 decile sort와 long-short portfolio 성과, macro regime별 차이를 본다. Lens의 IC/top-k/long-short/regime 평가 방향과 맞다.
- [Advances in Financial Machine Learning, O'Reilly 목차](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/ftoc.xhtml)
  - 핵심 요약: purged CV, embargo, backtest statistics, strategy risk 등 금융 ML 평가에서 누수와 과최적화 방어가 중요함을 보여준다.
- [mlfinlab Purged and Embargo Cross Validation 문서](https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html)
  - 핵심 요약: 금융 cross-validation에서는 label 기간이 겹치는 train sample 제거와 test 이후 embargo가 필요하다. Lens split의 h_max gap 철학과 같은 방향이다.
- [LightGBM parameters, ranking objective](https://lightgbm.readthedocs.io/en/v4.5.0/Parameters.html)
  - 핵심 요약: `lambdarank`, `rank_xendcg` 같은 ranking objective가 있다. Phase 1에서는 모델로 추가하지 말고 rank metric과 target 설계 참고용으로만 둔다.
- [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking.html)
  - 핵심 요약: Pairwise, groupwise ranking loss와 NDCG 계열 평가를 제공한다. Lens에는 날짜별 cross-section rank 평가 아이디어만 차용한다.

