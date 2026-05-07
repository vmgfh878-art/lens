# CP148-LM 1D 보수적 예측선 연구 설계서

작성일: 2026-05-07  
역할: LM 에이전트  
범위: 연구 설계 문서 작성 전용. 모델 학습, W&B/Optuna 실행, DB/Supabase write, inference 저장, 프론트 수정, 데이터 백필/동기화, 코드 수정 없음.

## 1. 연구 배경

주가 수익률 예측은 노이즈가 크고, 단일 시점의 point forecast 정확도만으로는 투자 판단에 충분하지 않다. 일반적인 return prediction은 상승 기회와 하방 위험을 동시에 다루기 어렵고, 특히 모델이 실제로 위험한 구간을 안전하게 보이게 만드는 false-safe 오류는 제품 신뢰도를 크게 훼손한다.

Lens line layer의 목표는 낙관적인 평균 수익 예측선이 아니라 `line_series`로 표시되는 하방 보수적 예측선이다. 따라서 CP148-LM의 핵심은 가격 자체를 정확히 맞히는 것이 아니라 500티커 EODHD cross-sectional universe에서 종목 간 상대 순위와 하방 위험 회피 신호를 동시에 검증하는 것이다.

1D h5는 단기 매매와 위험 점검에서 가장 실제적인 기본 단위다. CP146 500티커 baseline full training에서는 1D PatchTST 후보들이 IC/spread는 양수였지만 false-safe와 severe downside recall 기준을 통과하지 못했다. 그러므로 CP148은 단순 튜닝이 아니라 실패 원인을 분류할 수 있는 연구형 실험 프로토콜로 설계한다.

## 2. 기존 근거 요약

### CP145/CP148-0 데이터 전제

| 항목 | 값 |
|---|---:|
| universe | EODHD S&P500 503 ticker |
| 1D price rows | 1,387,834 |
| 1D indicator rows | 1,355,956 |
| price date range | 2015-01-02 ~ 2026-05-05 |
| 1D indicator date range | 2015-03-30 ~ 2026-05-05 |
| provider/source | eodhd / eodhd |
| feature contract | v3_adjusted_ohlc |
| model feature count | 36 |
| atr_ratio model feature 포함 여부 | false |
| context checksum | 1aa6452d82369cc6 |
| 1D source_data_hash, CP146 기준 | 569f21aa0cab3925 |
| 1D eligible ticker count, LM h5 | 473 |
| 1D split train/val/test rows | 799,739 / 171,363 / 171,837 |
| split overlap | 0 |
| feature NaN/Inf | 0 |
| target h5 NaN/Inf | 0 |

`full_features`는 macro/breadth 값은 실제로 채워져 있지만 fundamentals coverage가 낮다. CP148-0 기준 1D `has_fundamentals` true rate는 약 1.14%이고, company fundamentals ticker coverage는 98/503이다. 따라서 full_features는 기본 후보가 아니라 WARN 비교 후보로만 둔다.

### CP146 1D 결과 해석

| 후보 | val IC | val spread | val false_safe_tail | val severe_recall | 판정 |
|---|---:|---:|---:|---:|---|
| PatchTST h5 pvv p32/s16 beta2 | 0.0531 | 0.0072 | 0.3118 | 0.6842 | rejected |
| PatchTST h5 no_fundamentals p32/s16 beta2 | 0.0526 | 0.0092 | 0.3020 | 0.6889 | rejected |
| PatchTST h5 pvv dense beta2 | 0.0825 | 0.0096 | 0.3524 | 0.6439 | rejected |

해석은 명확하다. 1D line은 수익 순위 신호는 생겼지만, 보수적 위험선으로 보기에는 false_safe_tail이 높고 severe_downside_recall이 낮다. 따라서 CP148의 연구 목표는 IC/spread를 유지하면서 false-safe를 줄이는 구조, feature set, regularization 조합을 찾는 것이다.

## 3. 연구 질문

RQ1. 500티커 cross-sectional universe에서 deep time-series model이 baseline보다 의미 있는 ranking signal을 만들 수 있는가?

RQ2. 보수적 loss 설계가 false-safe downside signal을 줄이면서도 수익 추종성을 유지할 수 있는가?

RQ3. PatchTST, TiDE, CNN-LSTM, TCN 중 어떤 구조가 1D h5 line forecasting에 가장 적합한가?

RQ4. feature set은 가격/변동성/거래량 중심이 좋은가, context 확장 feature가 좋은가?

RQ5. validation 기준으로 고른 모델이 test 및 seed stability에서도 유지되는가?

## 4. 가설

H1. PatchTST는 긴 daily context를 patch로 압축해 pvv feature에서 가장 안정적인 후보가 될 것이다.

H2. no_fundamentals는 낮은 fundamentals coverage로 인한 잡음을 줄여 full_features보다 안정적일 수 있다.

H3. TiDE는 빠르고 dense한 covariate 처리에 유리하지만, coverage가 불균등한 context에서는 false-safe가 높아질 수 있다.

H4. TCN은 국소 temporal pattern과 최근 변동성에 강해 CNN-LSTM보다 안정적인 대안이 될 수 있다.

H5. beta=2.0 보수성은 false-safe를 줄이지만, 지나친 beta 조정은 upside_sacrifice를 키우므로 이번 실험에서는 고정한다.

## 5. 데이터셋과 계약

| 항목 | 계약 |
|---|---|
| timeframe | 1D |
| horizon | h5 |
| universe | EODHD S&P500 503 ticker, LM eligible 473 |
| provider/source | eodhd |
| feature contract | v3_adjusted_ohlc |
| target | raw_future_return |
| line_target_type | raw_future_return |
| checkpoint_selection | line_gate |
| line loss | AsymmetricHuberLoss alpha=1.0, beta=2.0, delta=1.0 |
| absolute minimum history gate | 1D 450 rows |
| split | time ordered train/val/test, purge gap 유지 |
| product 저장 | 별도 CP에서만 수행 |

1M은 이번 연구에서 제외한다. 1M은 데이터 밀도가 낮고 단기 product line v1의 우선순위가 아니며, h5 daily line의 실패 원인을 분리하기 전에 timeframe을 넓히면 해석이 흐려진다.

yfinance와 EODHD는 섞지 않는다. CP148은 EODHD 500 local parquet 기준의 제품 스케일 연구이며, yfinance/generic cache를 사용하면 provider/source/cache 계약이 깨진다.

## 6. 비교 모델

### PatchTST

PatchTST는 긴 daily context를 patch 단위로 압축하는 구조다. 1D h5에서 기대하는 장점은 장기 추세와 최근 변동성 패턴을 함께 보면서 cross-sectional ranking을 만들 수 있다는 점이다. 실패 가능성은 dense overlap이나 큰 capacity가 false-safe를 줄이기보다 위험한 구간에서 과도하게 낙관적인 순위 신호를 만들 수 있다는 점이다. 강해야 할 metric은 IC, long_short_spread, fee_adjusted_return이며, 제품 후보가 되려면 false_safe_tail과 severe_downside_recall도 동시에 통과해야 한다.

### TiDE

TiDE는 covariate 처리와 빠른 학습에 유리하다. pvv처럼 compact한 feature set에서는 짧은 실험 주기로 구조 비교가 가능하다. 실패 가능성은 context coverage가 불균등할 때 missingness나 regime feature를 부정확하게 해석해 false-safe가 커지는 것이다. 강해야 할 metric은 spread_ir, fee_adjusted_return, false_safe_tail 안정성이다.

### CNN-LSTM

CNN-LSTM은 국소 패턴 추출과 순차 기억을 결합한다. 1D line에서는 최근 수익률, 변동성, 거래량 충격 같은 국소 구조를 잡는 데 기대가 있다. 실패 가능성은 긴 seq_len 252에서 PatchTST보다 장기 의존성 처리 효율이 떨어지거나, 종목 간 ranking signal이 약해지는 것이다. 강해야 할 metric은 direction_accuracy, false_safe_tail, severe_downside_recall이다.

### TCN

TCN은 dilated temporal convolution으로 최근부터 중기까지의 temporal pattern을 안정적으로 볼 수 있다. LSTM보다 병렬화와 gradient 흐름이 유리할 수 있어 CNN-LSTM의 대안으로 둔다. 실패 가능성은 구현이 quantile/band 중심 skeleton이면 line head 계약이 충분히 검증되지 않았을 수 있다는 점이다. 강해야 할 metric은 false_safe_tail, severe_downside_recall, validation/test gap 안정성이다.

## 7. Baseline 설계

딥러닝 모델 전 반드시 baseline을 둔다. baseline의 목적은 딥러닝이 실제 signal을 배웠는지, IC/spread가 우연인지, false_safe 기준점이 어느 정도인지 확인하는 것이다.

| baseline | 의미 | 주요 확인 |
|---|---|---|
| zero return | 모든 미래 수익률을 0으로 보는 기준 | false-safe와 direction 기준점 |
| historical mean | 과거 평균 수익률 기반 | 단순 종목별 편향 대비 |
| 5D momentum | 최근 5일 수익률 추종 | 단기 추세 대비 |
| 20D momentum | 최근 20일 수익률 추종 | 중기 추세 대비 |
| short reversal | 최근 하락/상승 반전 가정 | mean reversion 대비 |
| rolling mean | rolling window 평균 | 안정적 평균선 대비 |
| existing 1D product line | 기존 저장 후보 `patchtst-1D-efad3c29d803` | 제품 후보 대비 개선 |
| random/shuffled baseline | 가능하면 seed 고정 셔플 | 우연 IC/spread 판별 |

각 baseline은 IC, long_short_spread, fee_adjusted_return, false_safe_tail, severe_downside_recall, direction_accuracy를 산출해야 한다.

## 8. Feature Set 설계

| feature_set | 포함 feature | 제외 feature | 기대 효과 | 위험 | 제품 설명 가능성 |
|---|---|---|---|---|---|
| price_volatility_volume | log_return, OHLC ratio, vol_change, MA ratio, RSI, MACD, BB position | macro, breadth, regime, fundamentals, calendar | 가장 해석 가능한 최소 feature. CP146의 1D ranking signal 기준점 | context 부재로 market-wide downside를 놓칠 수 있음 | 높음 |
| no_fundamentals | pvv + macro, breadth, regime, calendar, missingness flag | revenue, net_income, equity, eps, roe, debt_ratio | fundamentals low coverage 잡음 제거와 market context 유지 | context가 false-safe를 줄이지 못하면 feature noise | 중간~높음 |
| technical_only | 현재 정의상 pvv와 동일한 11개 feature | market context, fundamentals, calendar, missingness flag | 모델군별 technical-only 재현 기준 | pvv와 중복이면 별도 실행 가치 낮음 | 높음 |
| full_features | 전체 36개 | 없음 | context 전체 비교 기준 | fundamentals coverage가 낮아 해석 WARN | 낮음, WARN 비교만 |
| context_light | 미정의 | 미정의 | pvv + breadth/regime 정도의 light context 후보 | feature_set 정의 필요 | design_needed |

우선순위는 `price_volatility_volume`, `no_fundamentals`, `technical_only`, `full_features` 순서다. full_features는 기본 후보가 아니라 비교 후보로만 둔다.

## 9. 성공 기준

제품 v1 hard gate:

- line_gate pass
- IC > 0
- long_short_spread > 0
- fee_adjusted_return > 0
- false_safe_tail_rate <= 0.20
- severe_downside_recall >= 0.75
- source/provider/cache 명확
- seed stability 확인 전에는 provisional candidate

선호 기준:

- 기존 1D 모델보다 false_safe 개선
- baseline 3개 이상 능가
- upside_sacrifice가 과도하지 않음
- seed variance가 후보 간 성능 차이를 압도하지 않음
- 제품 설명이 가능한 최소 feature group 우선

## 10. 실험 단계

자세한 실행 절차는 [cp148_lm_1d_experiment_protocol.md](cp148_lm_1d_experiment_protocol.md)에 둔다.

요약:

1. Stage 0: data/cache gate
2. Stage 1: baseline
3. Stage 2: coarse model family comparison
4. Stage 3: feature set comparison
5. Stage 4: narrow Optuna/W&B sweep
6. Stage 5: seed stability
7. Stage 6: final candidate selection
8. Stage 7: product save는 별도 CP

## 11. 결과 해석 원칙

MAE/SMAPE는 보조 지표다. 이 연구의 핵심은 IC, long_short_spread, fee_adjusted_return, false_safe_tail_rate, severe_downside_recall이다.

IC/spread가 높아도 false_safe_tail이 0.20을 넘고 severe_downside_recall이 0.75 미만이면 제품 v1 후보가 아니다. 반대로 IC가 약하더라도 false_safe와 severe recall이 매우 좋으면 risk-only watch로 기록할 수 있지만, 1D 제품 기본 line은 ranking과 downside 방어를 모두 만족해야 한다.

validation 기준으로 후보를 고르고, test는 최종 참고와 overfit 진단에만 사용한다. test에서만 좋은 후보는 제품 후보로 올리지 않는다.

## 12. 사용자 승인 체크리스트

실험 실행 전 사용자가 승인해야 할 항목:

- h5만 먼저 실행할지, h10/h20은 별도 CP로 미룰지
- beta=2.0을 이번 연구 전체에서 고정할지
- full_features를 WARN 비교 후보로 1회만 둘지
- TCN line을 1차 coarse 후보에 포함할지
- Optuna trial budget
- seed 재실행 개수
- W&B online 사용 여부와 project 이름

## 13. 판정

현재 설계 판정: PASS.

이 문서만으로 CP148-LM-1D 실행 지시서 초안으로 변환 가능하다. 다만 `context_light`는 feature_set 정의가 없으므로 design_needed이며, TCN은 preflight forward/loss는 통과했지만 line product 후보로서의 경험치가 적어 coarse 비교에서 실험 기록 성격이 강하다.
