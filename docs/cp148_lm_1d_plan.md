# CP148-LM 1D 실험 계획서 REV4

한 줄 목적: EODHD 500티커 기준 1D h5 보수적 예측선 v1을 만들기 위한 연구형 실험 계획.

이 문서 하나를 실험 직전 기준 문서로 사용한다. 기존 `cp148_lm_1d_research_design.md`, `cp148_lm_1d_experiment_protocol.md`, `cp148_lm_1d_metric_dictionary.md`, `cp148_lm_1d_failure_taxonomy.md`, `cp148_lm_1d_experiment_matrix.csv`는 삭제하지 않고 부록/원자료로만 둔다.

## 1. 왜 이 실험을 하는가

100티커 모델은 초기 검증과 데모 후보였다. 제품 후보 최소 기준은 EODHD 500티커 규모로 올린다.

현재 1D line은 EODHD 500티커 기준 제품 v1이 없다. CP146에서 1D PatchTST 후보들은 IC와 spread는 양수였지만 false_safe_tail_rate가 0.30 이상, severe_downside_recall이 0.69 미만이라 보수적 예측선 제품 목표치를 통과하지 못했다.

목표는 가격을 정확히 맞히는 것이 아니다. 1D h5 기준으로 종목 간 상대 순위를 만들고, 실제 하방 위험을 안전하게 보이는 false-safe 오류를 줄이는 보수적 예측선을 찾는 것이다.

## 2. Line과 Band 역할

이번 CP148-LM-1D는 line 실험이다. band 실험과 분리한다.

| 구분 | 의미 | 이번 CP 사용 여부 |
|---|---|---|
| 1D line | beta=2.0 asymmetric loss를 쓰는 보수적 point forecast | 사용 |
| lower quantile | quantile interval의 하단 | 사용 안 함 |
| band | lower/upper quantile interval | 사용 안 함 |
| composite/overlay | line과 band를 합친 후처리 표시 또는 평가 | 사용 안 함 |

1D line은 lower quantile이 아니다. 모델이 하나의 평균선을 내는 것도 아니고, beta=2.0 asymmetric loss로 과대예측을 더 강하게 벌주는 trained conservative point forecast다. band는 별도의 quantile interval이며, coverage/lower breach/upper breach는 이번 line 후보 선별 기준이 아니다.

## 3. 데이터 전제

| 항목 | 기준 |
|---|---|
| provider/source | eodhd / eodhd |
| 원천 universe | EODHD S&P500 503 ticker |
| 1D LM eligible | 473 ticker |
| timeframe | 1D |
| horizon | h5 |
| feature contract | v3_adjusted_ohlc |
| source_data_hash, CP146 1D 기준 | 569f21aa0cab3925 |
| context checksum | 1aa6452d82369cc6 |
| absolute minimum | 450 rows |
| split | time ordered train/val/test |
| train/val/test rows | 799,739 / 171,363 / 171,837 |
| purge gap / overlap | PASS / 0 |
| feature NaN/Inf | 0 |
| target h5 NaN/Inf | 0 |
| model features | 36 |
| atr_ratio model feature 포함 | false |

`full_features`는 fundamentals coverage가 낮다. 1D `has_fundamentals` true rate는 약 1.14%이고 company fundamentals ticker coverage는 98/503이다. 따라서 `full_features`는 기본 후보가 아니라 WARN 비교 후보로만 둔다.

## 4. 기존 1D Product Line Artifact

기존 1D product line은 baseline/reference로만 사용한다. 500티커 제품 후보 기준과 혼동하지 않는다.

| 항목 | 값 |
|---|---|
| run_id | patchtst-1D-efad3c29d803 |
| checkpoint | C:\Users\user\lens\ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-efad3c29d803.pt |
| role | line_model |
| timeframe/horizon | 1D / h5 |
| seq_len / patch | 252 / p32/s16 |
| feature_set | full_features |
| beta | 2.0 |
| 사용 목적 | existing 1D product line baseline/reference |
| 주의 | CP75 데이터 hash 기준 저장 후보이며, CP148 EODHD 500 제품 v1 후보와 동일시하지 않음 |

## 5. 이미 결정한 것

사용자가 다시 승인하지 않아도 되는 결정사항이다.

| 항목 | 결정 |
|---|---|
| beta | 2.0 고정 |
| alpha / delta | 1.0 / 1.0 |
| target | raw_future_return |
| line_target_type | raw_future_return |
| checkpoint_selection | line_gate |
| provider/source | eodhd |
| horizon | h5 중심 |
| fee assumption | 10bp 고정 |
| final validation | 최종 2개 후보만 3-fold walk-forward + seed stability |
| save-run | 이번 CP에서는 없음 |
| display calibration | 없음 |
| 1W/1M | 제외 |
| band/composite | 제외 |
| inference 저장 | 없음 |

beta=2.0은 Lens line의 학습 보수성 계약이다. beta 변경은 별도 loss study CP로 분리한다.

## 6. Horizon 정책

이번 CP148은 h5를 깊게 판다. h1 전용 제품 슬롯은 열지 않는다.

h1은 제품 슬롯이 아니라 h1~h5 path consistency와 단기 보조 진단으로만 확인한다. 예를 들어 h5 line이 좋은데 h1~h3 경로가 완전히 반대로 움직이면 표시 안정성 WARN을 줄 수 있다. h1 전용 제품 실험은 후속 CP로 분리한다.

## 7. 주요 Metric 산식

아래 산식은 CP148-LM-1D 후보 선택에 쓰는 기본 정의다. 모든 metric은 validation 중심으로 계산하고, test는 최종 참고와 overfit 진단에만 쓴다.

| 지표 | 산식 | 해석 |
|---|---|---|
| IC | 날짜별 cross-sectional Spearman correlation을 구한 뒤 시계열 평균 | 같은 날짜에 어떤 종목이 상대적으로 좋은지 맞히는 ranking signal |
| ic_std | 날짜별 IC 시계열의 표준편차 | IC 안정성의 분모 |
| ic_ir | ic_mean / ic_std | IC 평균 대비 변동성 |
| ic_t_stat | ic_mean / (ic_std / sqrt(number_of_dates)) | IC가 0보다 큰지 보는 참고값 |
| long_short_spread | 날짜별 예측 상위 basket 실제 수익률 평균 - 예측 하위 basket 실제 수익률 평균, basket 내부 equal weight | ranking이 실제 수익 차이로 이어지는지 확인 |
| fee_adjusted_return | long_short_return - turnover * 10bp | 10bp 고정 비용 차감 후 ranking signal이 남는지 보는 보조 지표 |
| false_safe_tail_rate | count(actual_return <= tail_threshold and prediction > safe_threshold) / count(actual_return <= tail_threshold) | tail downside를 안전하게 보인 비율 |
| severe_downside_recall | count(actual_return <= severe_threshold and prediction <= safe_threshold) / count(actual_return <= severe_threshold) | severe downside 분모 중 위험으로 포착한 비율 |
| conservative_bias | mean(prediction - actual_return) | 음수면 평균적으로 보수적 |
| upside_sacrifice | mean(max(actual_return - prediction, 0) for actual_return > 0) | 상승 구간에서 보수 예측이 희생한 평균 폭 |

Tail/safe/severe threshold는 Stage 1 baseline 산출 시 split별 target 분포를 보고 고정한다. 기본 원칙은 tail/severe downside를 validation 분포의 하위 quantile로 정의하고, safe 판정은 line이 위험하지 않다고 표시한 경우로 둔다. threshold를 test를 보고 바꾸지 않는다.

`fee_adjusted_return`은 포트폴리오 전략 최적화 지표가 아니다. 날짜별 top/bottom basket, equal weight, 매 평가일 rebalance, turnover 10bp 비용 차감을 둔 ranking signal 내구성 보조 지표다.

## 8. 이번 실험에서 열어볼 것

| 축 | 후보 |
|---|---|
| model family | PatchTST, TiDE, CNN-LSTM |
| feature_set | price_volatility_volume, no_fundamentals, full_features WARN 비교 |
| PatchTST patch | p32/s16, p16/s8 |
| seq_len | 252 기본, 180 또는 120 제한 비교 |
| narrow sweep 후보 | lr, dropout, weight_decay, batch_size |

## 9. TCN 상태와 BM 이관

CP148-0-S-TCN-AUDIT 결과는 WARN으로 기록한다. causal leakage, output shape, loss 계약은 통과했지만, 현재 receptive field가 61/252 수준이라 1D h5 line의 장기 문맥 주력 후보로는 약하다.

따라서 TCN은 CP148-LM-1D Stage 2 coarse 후보에서 제외한다. TCN은 폐기하지 않고 1D/1W band 실험에서 변동성, 국소 패턴, 밴드 폭, 하방 위험 모델 후보로 재검토한다.

정리: TCN은 line exploratory 후보가 아니라 BM 후보로 이관한다.

## 10. Baseline

| baseline | 목적 |
|---|---|
| zero return | 모든 예측을 0으로 놓고 false-safe와 direction 기준점을 만든다 |
| historical mean | 종목별 장기 평균만으로 생기는 단순 편향을 확인한다 |
| 5D momentum | 짧은 추세 추종 기준을 둔다 |
| 20D momentum | 중기 추세 추종 기준을 둔다 |
| short reversal | 단기 반전 가정이 h5에서 통하는지 확인한다 |
| rolling mean | 최근 rolling 평균선이 만드는 안정적 기준을 본다 |
| 기존 1D product line | 기존 저장 후보 대비 false-safe 개선 여부를 본다 |
| random/shuffled | IC/spread가 우연인지 판별한다. 가능할 때 seed 고정으로 실행한다 |

Stage 1 baseline 결과를 본 뒤 minimum candidate gate와 product target을 분리해 재정렬한다.

## 11. Baseline 이후 Gate 재정렬

최초 제품 목표치는 다음과 같다.

- line_gate pass
- IC > 0
- long_short_spread > 0
- fee_adjusted_return > 0
- false_safe_tail_rate <= 0.20
- severe_downside_recall >= 0.75

단, 이 값을 처음부터 모든 후보의 절대 hard gate로 고정하지 않는다. Stage 1 baseline 결과를 본 뒤 다음 두 층을 분리해 확정한다.

| 층 | 의미 | 확정 시점 |
|---|---|---|
| minimum candidate gate | baseline 대비 의미 있는 후보를 Stage 2/3로 넘기는 최소 기준 | Stage 1 baseline 이후 |
| product target | 제품 v1 승격 목표치. false_safe <= 0.20, severe_recall >= 0.75는 여기에 둔다 | Stage 1 이후 문구 재확인, 최종 promotion까지 유지 |

예를 들어 baseline이 이미 false_safe_tail 0.30 근처라면 Stage 2에서는 baseline 대비 개선한 후보를 sweep 후보로 남길 수 있다. 그러나 제품 v1로 승격하려면 product target을 통과해야 한다.

## 12. 1차 Coarse 실험 목록

| experiment_id | model | feature_set | seq_len | horizon | 주요 hp | 왜 하는가 | 기대 장점 | 실패 가능성 | 다음 결정 |
|---|---|---|---:|---:|---|---|---|---|---|
| cp148_s2_patchtst_pvv_p32_s16 | PatchTST | price_volatility_volume | 252 | 5 | patch32/stride16, beta2 | CP146 pvv 기준 재검증 | 긴 context ranking | false_safe_fail 반복 | 통과하면 top 후보 |
| cp148_s2_patchtst_no_fund_p32_s16 | PatchTST | no_fundamentals | 252 | 5 | patch32/stride16, beta2 | fundamentals 제거 효과 확인 | market context 유지, 잡음 감소 | context noise | pvv 대비 개선이면 feature 확장 후보 |
| cp148_s2_patchtst_pvv_p16_s8 | PatchTST | price_volatility_volume | 252 | 5 | patch16/stride8, beta2 | dense patch가 하방 포착을 돕는지 확인 | 최근 변동성 반응 | IC는 좋아도 false-safe 악화 | risk 지표 개선 없으면 탈락 |
| cp148_s2_patchtst_pvv_seq180_or_120 | PatchTST | price_volatility_volume | 180 또는 120 | 5 | patch16/stride8, beta2 | 짧은 context가 false-safe를 줄이는지 확인 | 최근성 강화, 빠른 실험 | 장기 ranking 손실 | seq252보다 risk 개선 시 sweep 후보 |
| cp148_s2_tide_pvv | TiDE | price_volatility_volume | 252 | 5 | beta2 | covariate dense model 비교 | 빠른 학습, 구조 대안 | false-safe 높아질 수 있음 | PatchTST 대비 risk 개선 시 유지 |
| cp148_s2_cnn_lstm_pvv | CNN-LSTM | price_volatility_volume | 252 | 5 | beta2 | 국소 패턴 + 순차 기억 비교 | 최근 변동/거래량 충격 포착 | long context 효율 저하 | direction/risk 강하면 유지 |

## 13. Top 후보 선정 Composite Score

top 2~3 후보는 사람이 감으로 고르지 않는다. baseline/gate 통과 여부와 composite score를 함께 본다.

Composite score는 validation 기준 rank score로 계산한다. 높은 순위가 좋은 지표는 그대로, 낮을수록 좋은 지표는 역순위를 쓴다.

| 항목 | 가중치 |
|---|---:|
| IC rank | 0.20 |
| spread rank | 0.15 |
| fee_adjusted_return rank | 0.15 |
| false_safe_tail 낮음 rank | 0.25 |
| severe_recall rank | 0.20 |
| upside_sacrifice 낮음 rank | 0.05 |

선정 순서:

1. cache/provider/metric 생성 실패 후보 제거
2. baseline 대비 완전 열위 후보 제거
3. minimum candidate gate 통과 여부 표시
4. composite score로 정렬
5. top family가 한 구조에 몰리면 구조 다양성을 위해 같은 family 중복을 제한할 수 있음

composite score는 후보 선정 보조 기준이며, checkpoint_selection은 계속 line_gate를 사용한다.

## 14. Failure Taxonomy와 조정 방향

| 실패 유형 | 뜻 | 다음 실험 조정 |
|---|---|---|
| ranking_fail | IC <= 0 | model family 변경, seq_len 축소, pvv/no_fundamentals 재비교 |
| false_safe_fail | false_safe_tail 높음 | noisy full feature 축소, dropout/weight_decay 강화, pvv/technical 우선, beta 변경은 별도 CP |
| severe_recall_fail | severe_downside_recall 낮음 | downside sample weighting 또는 false-safe-aware selector를 다음 CP로 제안 |
| over_conservative_fail | line이 과도하게 낮아 수익 추종성 붕괴 | risk-only로만 기록, beta/offset 조정은 별도 CP |
| fee_negative_fail | fee_adjusted_return <= 0 | turnover/rebalance 민감도 점검, ranking은 좋지만 비용 취약으로 분류 |
| val_test_gap | validation은 좋고 test에서 붕괴 | walk-forward 후보 제외 또는 seed/기간 안정성 재검증 |
| seed_unstable | seed별 결론이 바뀜 | seed median 기준으로 재정렬, 최종 후보 제외 가능 |
| feature_noise_fail | feature 확장 후 pvv보다 악화 | no_fundamentals/pvv 우선, full_features WARN 유지 |
| cache_contract_fail | provider/source/cache 불명확 | 실험 폐기, Stage 0 재실행 |

## 15. Stage별 시간/메모리 견적

실행 전 Stage 0에서 50티커 1epoch timing smoke를 먼저 측정하고, 그 값을 기준으로 500티커 시간을 갱신한다.

추정식:

- 500티커 1epoch 예상 시간 = 50티커 1epoch 측정 시간 * 10 * overhead_factor
- overhead_factor 기본값 = 1.2~1.5
- 500티커 총 시간 = 500티커 1epoch 예상 시간 * epochs * seeds * trials
- VRAM은 50티커 측정값과 CP146/CP75 실제 기록을 함께 본다.

참고 기록:

| 근거 | 값 |
|---|---|
| CP146 PatchTST 1D pvv p32/s16 473티커 5epoch | 1,099.56초, 약 219.91초/epoch |
| CP146 PatchTST 1D no_fundamentals 473티커 5epoch | 2,082.43초, 약 416.49초/epoch |
| CP146 PatchTST 1D pvv dense 473티커 5epoch | 1,567.70초, 약 313.54초/epoch |
| CP75 PatchTST 1D full_features 473티커 5epoch | 2,810.04초, peak VRAM 약 3,030.66MB |

운영 견적:

| Stage | 예상 후보/trial | epoch | seed | 예상 시간 | 예상 VRAM | stop rule |
|---|---:|---:|---:|---|---|---|
| Stage 1 baseline | 8 baseline | n/a | n/a | 수분~1시간 | CPU 중심 | baseline metric 생성 실패 시 중단 |
| Stage 2 coarse | 6 후보 | 3~5 | 가능하면 3 | 50티커 1epoch 측정값으로 갱신. CP146 기준 단일 seed 20~45분/후보 | PatchTST 기준 3~6GB 예상 | 2주 내 top family가 안 나오면 후보 축소 |
| Stage 3 narrow sweep | top 2~3 family, 8~24 trials | 3~5 | 1, 상위 config는 3 | 보통 예산 기준 1~3주 | 3~8GB 예상 | false_safe 개선 없는 trial이 70% 이상이면 중단 |
| Stage 4 seed/walk-forward | 최종 2 후보 | 5 | seed 5, 3-fold | 1~2주 | 후보별 측정값 사용 | 6주 예산 초과 전 product target 미근접이면 연구 보류 |

전체 연구는 6주 예산 안에서 끊는다. 4주차까지 baseline 대비 false_safe 개선 후보가 없으면 Stage 3 확대를 멈추고 실패 taxonomy를 확정한다. 6주차까지 product target 후보가 없으면 제품 저장 CP로 넘기지 않는다.

## 16. 실험 단계

| Stage | 입력 | 출력 | 중단 조건 | 다음 단계 조건 |
|---|---|---|---|---|
| Stage 0: data/cache preflight | EODHD 1D price/indicator/context, cache manifest, 50티커 1epoch timing smoke | source hash, eligible count, finite, split, 500티커 시간/VRAM 갱신 | cache/provider 혼입, feature/target nonfinite, split overlap | 모든 gate PASS |
| Stage 1: baseline | h5 target, train/val/test split | baseline metric table, minimum gate 재정렬 | baseline metric 생성 실패 | baseline 기준선 확보 |
| Stage 2: coarse model 비교 | 1차 후보, 가능하면 seed 3개 | seed median line_metrics, composite score | 모든 후보 ranking/risk 붕괴 | top family 2~3개 선별 |
| Stage 3: narrow Optuna/W&B sweep | Stage 2 top family | 작은 hp 탐색 결과 | false-safe 지속 악화, metric NaN/Inf | 상위 config seed 3개 재평가 |
| Stage 4: seed stability | sweep best config | seed 3개 재평가, 최종 2개 후보 선별 | seed variance가 성능 차이를 압도 | final 2 후보만 walk-forward |
| Stage 5: walk-forward + seed stability | 최종 2개 후보 | seed 5개 + 3-fold walk-forward | val/test/walk-forward 붕괴 | product promotion 후보 결정 |
| Stage 6: 저장/프론트 연결 | Stage 5 승인 후보 | 별도 CP에서만 save-run/product 연결 | 이번 CP 범위 아님 | 사용자 명시 승인 필요 |

모든 후보에 3-fold walk-forward를 걸지 않는다. 최종 큰 후보 2개에만 3-fold walk-forward와 seed stability를 적용한다. 이 절차를 product promotion 조건으로 둔다.

## 17. Seed와 Walk-forward 순서

Coarse stage는 가능하면 seed 3개 median으로 판단한다. 시간이 부족하면 seed 1개 coarse를 허용하되, top family 선택 전에 최소 seed 3개 재평가를 권장한다.

Optuna/W&B sweep은 top family 중심으로 한다. sweep의 best trial은 결론이 아니며, 상위 config는 seed 3개로 재평가한다.

최종 2개 후보는 seed 5개 재실행과 3-fold walk-forward를 적용한다. 평균뿐 아니라 worst seed와 worst fold가 product target을 크게 벗어나지 않는지 확인한다. 이 절차를 통과해야 product promotion 후보가 된다.

## 18. Optuna/W&B 운영

처음부터 전체 sweep 금지. coarse 비교 후 top 2~3개 family만 narrow sweep한다.

W&B online은 사용자가 직접 터미널에서 켠다. W&B init 실패 시 학습 자체는 local log 기준으로 계속 가능해야 한다.

Optuna trial budget 제안:

| 예산 | trial 수 | 용도 |
|---|---:|---|
| 작은 예산 | 8~12 | top 후보가 너무 불확실할 때 빠른 방향 확인 |
| 보통 예산 | 20~24 | 기본 추천. 비용과 탐색 균형 |
| 넉넉한 예산 | 36~48 | coarse 결과가 좋고 제품 후보 가능성이 높을 때 |

best trial은 seed 3~5개 재실행 전까지 결론이 아니다.

## 19. 사용자가 승인할 항목

승인 항목은 실제 선택이 필요한 것만 남긴다.

- [ ] Optuna trial budget을 고른다: 작은 예산 / 보통 예산 / 넉넉한 예산.
- [ ] seed 재실행 개수를 고른다: coarse seed 3개 권장, 최종 seed 5개 권장.
- [ ] W&B online 사용 여부를 정한다.

이미 결정된 항목:

- h5 중심
- beta=2.0 고정
- fee 10bp 고정
- full_features는 WARN 비교 후보
- TCN은 line 후보에서 제외하고 BM 후보로 이관
- final 2개 후보만 3-fold walk-forward
- band/composite/save-run/inference 저장 제외

## 20. 바로 다음 실행 지시서 초안

아래는 실행 오더로 바꿀 수 있는 초안이다. 실제 실행은 사용자 승인 전 금지한다.

```text
CP148-LM-1D 실행 지시서 초안

목표:
EODHD 500 local parquet 기준 1D h5 line_model 후보를 baseline -> coarse model 비교 -> top 후보 narrow sweep -> final 2 walk-forward 순서로 검증한다.

금지:
save-run 금지, DB write 금지, inference 저장 금지, band/composite 금지, 프론트 수정 금지, yfinance/EODHD live fetch 금지.

공통:
provider/source=eodhd
timeframe=1D
horizon=5
target=raw_future_return
line_target_type=raw_future_return
checkpoint_selection=line_gate
alpha/beta/delta=1.0/2.0/1.0
fee_assumption=10bp
feature_contract=v3_adjusted_ohlc

Stage 0:
EODHD cache/source/finite/split preflight 확인.
50티커 1epoch timing smoke로 500티커 시간/VRAM 추정 갱신.

Stage 1:
zero, historical mean, 5D momentum, 20D momentum, short reversal, rolling mean, existing 1D product line, random/shuffled baseline 평가.
baseline 이후 minimum candidate gate와 product target 분리.

Stage 2:
PatchTST pvv p32/s16
PatchTST no_fundamentals p32/s16
PatchTST pvv p16/s8
PatchTST pvv seq180 또는 seq120
TiDE pvv
CNN-LSTM pvv

Top 후보 선정:
baseline/gate 통과 여부와 validation composite score를 함께 사용.

평가:
IC, ic_std, ic_ir, ic_t_stat, long_short_spread, fee_adjusted_return, false_safe_tail_rate, severe_downside_recall, conservative_bias, upside_sacrifice, direction_accuracy, line_gate_pass.

판정:
coarse는 가능하면 seed 3개 median. sweep best는 seed 3개 재평가. 최종 2개 후보만 seed 5개와 3-fold walk-forward 확인.
```

## 21. 변경 이력

- REV3: TCN을 CP148-LM-1D line Stage 2 후보에서 제외하고, CP148-0-S-TCN-AUDIT WARN 근거와 함께 1D/1W BM 후보로 이관했다.
- REV4: metric 산식, baseline 이후 gate 재정렬, 시간/메모리 견적, composite score, seed/walk-forward product promotion 조건을 본문에 고정했다.

## 22. 부록/원자료

사용자가 읽을 기준 문서는 이 파일 하나다.

부록으로 남기는 기존 산출물:

- `docs/cp148_lm_1d_research_design.md`
- `docs/cp148_lm_1d_experiment_protocol.md`
- `docs/cp148_lm_1d_metric_dictionary.md`
- `docs/cp148_lm_1d_failure_taxonomy.md`
- `docs/cp148_lm_1d_experiment_matrix.csv`

판정: PASS 기준은 사용자가 `docs/cp148_lm_1d_plan.md` 하나만 읽고 실험 의도, 순서, 기준, 승인 항목을 이해할 수 있는 것이다.
