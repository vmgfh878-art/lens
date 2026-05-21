# CP63-BM AI band model 피처 감사 및 피처 그룹 실험 설계

## 1. 결론

이번 CP에서는 새 학습, save-run, DB 쓰기, DB schema 변경, UI/backend API 수정 없이 기존 1D feature cache만 읽어 AI band layer 피처 감사를 수행했다. 통계 baseline은 비교 기준으로만 남겼고, AI residual/scale 보정 모델은 설계하거나 구현하지 않았다.

BM 후보 우선순위는 다음과 같이 유지한다.

| 우선순위 | 후보 | CP63 해석 |
|---:|---|---|
| 1 | CNN-LSTM | `seq60 q15-b2 direct` 계열을 주 BM 후보로 두되, market/fundamental 노이즈를 줄인 `price_volatility` 세트를 먼저 본다. |
| 2 | TiDE | `q10-b2 param`은 watch, `q10-b2 direct`는 fail에 가까우므로 feature pruning smoke로만 확인한다. |
| 3 | PatchTST band | 현재 제품 band 주력 후보가 아니며 참고 후순위로만 둔다. |

감사 범위는 `feature_version=v3_adjusted_ohlc`, `target=raw_future_return`, `band_target_type=raw_future_return`, `role=band_model`, `checkpoint_selection=band_gate` 기준이다. 캐시는 `ai\cache\features_1D_1e47425bbc16_0c1d7f52.pt`를 읽었고, exact eligible scope는 50/100/188 ticker로 잘랐다.

## 2. 데이터 범위

| 항목 | 값 |
|---|---:|
| timeframe | 1D |
| seq_len | 60 |
| horizon | 5 |
| h_max split gap | 20 |
| input ticker | 188 |
| eligible ticker | 188 |
| scope eligible50 | 50 |
| scope eligible100 | 100 |
| scope eligible188 | 188 |

## 3. 36개 피처 인벤토리

| 피처 | 그룹 | band 관점 도움 가능성 | 약할 가능성 |
| --- | --- | --- | --- |
| log_return | price/return | 직전 수익률 충격은 다음 구간의 절대 변동성과 mean-reversion 위험을 설명할 수 있다. | 아니오 |
| open_ratio | price/return | 시가 갭은 단기 불확실성과 당일 정보 충격의 흔적이다. | 예 |
| high_ratio | volatility/range | 당일 상단 range는 가까운 미래의 band 상단/폭 확대 신호가 될 수 있다. | 아니오 |
| low_ratio | volatility/range | 당일 하단 range는 하방 꼬리와 lower breach 위험을 포착할 수 있다. | 아니오 |
| vol_change | volume/liquidity | 거래량 급증은 정보 유입과 다음 구간 변동성 확대의 proxy가 될 수 있다. | 예 |
| ma_5_ratio | momentum/oscillator | 초단기 이동평균 괴리는 squeeze 이후 반전 또는 확산 신호가 될 수 있다. | 아니오 |
| ma_20_ratio | momentum/oscillator | 20일 괴리는 중단기 과열과 band 폭 확대 가능성을 같이 담는다. | 아니오 |
| ma_60_ratio | momentum/oscillator | 60일 추세 괴리는 regime성 변동성 또는 추세 붕괴 위험을 설명할 수 있다. | 아니오 |
| rsi | momentum/oscillator | 과열/침체 위치는 하방 이벤트와 반전형 변동성의 proxy가 될 수 있다. | 아니오 |
| macd_ratio | momentum/oscillator | 추세 가속/둔화는 band 폭이 커질 regime을 구분하는 데 도움이 된다. | 아니오 |
| bb_position | momentum/oscillator | 볼린저 위치는 squeeze와 과열 위치를 동시에 담아 breach proxy로 쓸 수 있다. | 예 |
| us10y | market/sector/breadth | 금리 레벨은 전체 시장 변동성 regime을 설명할 수 있다. | 예 |
| yield_spread | market/sector/breadth | 장단기 금리차는 경기 regime과 risk-off 구간을 설명할 수 있다. | 예 |
| vix_close | market/sector/breadth | VIX는 다음 구간 절대 변동성과 가장 직접적인 시장 공포 proxy다. | 아니오 |
| credit_spread_hy | market/sector/breadth | 하이일드 스프레드는 stress regime의 하방 band 확대 신호가 될 수 있다. | 예 |
| nh_nl_index | market/sector/breadth | 신고가/신저가 폭은 시장 breadth 붕괴와 downside event를 포착할 수 있다. | 아니오 |
| ma200_pct | market/sector/breadth | 200일선 상회 비율은 넓은 시장 trend regime을 요약한다. | 예 |
| regime_calm | market/sector/breadth | calm regime은 좁은 band가 허용되는 구간을 알려줄 수 있다. | 예 |
| regime_neutral | market/sector/breadth | 중립 regime은 calm/stress 사이의 band 폭 보정값 역할을 할 수 있다. | 예 |
| regime_stress | market/sector/breadth | stress regime은 lower band 확대와 downside breach 방어에 직접 유리하다. | 아니오 |
| revenue | fundamentals | 기업 규모와 사업 안정성 proxy로 장기 변동성 차이를 설명할 수 있다. | 예 |
| net_income | fundamentals | 수익성은 재무 취약 구간의 downside risk를 설명할 수 있다. | 예 |
| equity | fundamentals | 자본 규모는 재무 buffer와 변동성 체질을 구분할 수 있다. | 예 |
| eps | fundamentals | 이익 수준은 하방 취약성과 이벤트 변동성의 약한 proxy가 될 수 있다. | 예 |
| roe | fundamentals | 자본 효율은 재무 품질과 하방 회복력을 나타낼 수 있다. | 예 |
| debt_ratio | fundamentals | 레버리지는 stress 구간 하방 band 확대 후보 신호다. | 예 |
| has_macro | missingness flag | macro 결측 처리 상태를 알려주지만 현재는 거의 상수다. | 예 |
| has_breadth | missingness flag | breadth 데이터 가용성 구간을 구분한다. | 예 |
| has_fundamentals | missingness flag | 재무 데이터 존재 여부가 규모/상장기간 proxy로 작동할 수 있다. | 예 |
| day_of_week_sin | calendar/future covariate | 요일별 변동성 패턴과 옵션/리밸런싱 효과를 약하게 담을 수 있다. | 예 |
| day_of_week_cos | calendar/future covariate | 요일 주기를 연속형으로 표현해 future covariate와 일관된 시간 정보를 준다. | 예 |
| month_sin | calendar/future covariate | 월중/계절성 변동성 패턴을 약하게 포착할 수 있다. | 예 |
| month_cos | calendar/future covariate | 월 계절성을 순환형으로 제공해 decoder future covariate와 맞물릴 수 있다. | 예 |
| is_month_end | calendar/future covariate | 월말 리밸런싱과 유동성 패턴의 band 확대 가능성을 알려준다. | 예 |
| is_quarter_end | calendar/future covariate | 분기말 리밸런싱/공시 시즌 근처 변동성 변화를 표현할 수 있다. | 예 |
| is_opex_friday | calendar/future covariate | 옵션 만기일 주변 변동성/유동성 변화를 반영할 수 있다. | 예 |

## 4. 피처 품질 감사 요약

현 36개 모델 입력은 exact eligible188 anchor sample 기준 non-null/finite ratio가 모두 1.0으로 통과했다. 다만 finite가 통과된다는 뜻이 band에 유용하다는 뜻은 아니다. 원시 재무값, missingness flag, 공통 macro/breadth 값, 희소 calendar flag는 정보량 또는 편향 측면에서 약할 수 있다.

train/val/test 평균 shift가 큰 피처는 다음과 같다.

| 피처 | 그룹 | train 평균 | val 평균 | test 평균 | max shift std |
| --- | --- | --- | --- | --- | --- |
| net_income | fundamentals | 1716633.2500 | 9001295.0000 | 550950976.0000 | 22.6559 |
| credit_spread_hy | market/sector/breadth | 0.0419 | 2.9497 | 3.0081 | 7.4067 |
| us10y | market/sector/breadth | 2.0846 | 4.0894 | 4.2375 | 2.7496 |
| has_fundamentals | missingness flag | 0.0181 | 0.0591 | 0.3362 | 2.3824 |
| revenue | fundamentals | 164546192.0000 | 586265664.0000 | 5625064960.0000 | 2.3387 |
| yield_spread | market/sector/breadth | 0.6479 | -0.4956 | 0.4261 | 2.2273 |
| regime_neutral | market/sector/breadth | 0.4396 | 0.4739 | 0.7931 | 0.7121 |
| regime_calm | market/sector/breadth | 0.3714 | 0.5166 | 0.1218 | 0.5166 |
| regime_stress | market/sector/breadth | 0.1890 | 0.0095 | 0.0852 | 0.4586 |
| vix_close | market/sector/breadth | 19.0500 | 15.9685 | 18.8256 | 0.3805 |

## 5. 밴드 예측에 유리한 top proxy 후보

학습 없이 계산한 proxy는 `abs_future_return_h5`, `downside_future_return_h5`, train q15/q85 기준 future band breach event, trailing 20일 realized volatility와의 관계만 사용했다. 이는 모델 성능 지표가 아니라 다음 실험 feature pruning을 위한 선별 신호다.

| 피처 | 모델 입력 | 그룹 | abs IC | downside IC | breach corr | rolling vol corr | proxy score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| atr_ratio | False | volatility/range | 0.3066 | 0.0602 | 0.2744 | 0.8925 | 0.3834 |
| intraday_range_ratio | False | volatility/range | 0.2578 | 0.0503 | 0.2335 | 0.5894 | 0.2828 |
| vix_close | True | market/sector/breadth | 0.2044 | 0.0208 | 0.1998 | 0.4365 | 0.2154 |
| regime_calm | True | market/sector/breadth | -0.1520 | -0.0208 | -0.1353 | -0.3281 | 0.1590 |
| regime_stress | True | market/sector/breadth | 0.1574 | 0.0050 | 0.1503 | 0.3164 | 0.1573 |
| ma200_pct | True | market/sector/breadth | -0.1419 | -0.0096 | -0.1142 | -0.2955 | 0.1403 |
| low_ratio | True | volatility/range | -0.1193 | -0.0201 | -0.1303 | -0.2047 | 0.1186 |
| nh_nl_index | True | market/sector/breadth | -0.1210 | 0.0028 | -0.1221 | -0.2002 | 0.1115 |
| high_ratio | True | volatility/range | 0.0759 | 0.0155 | 0.0893 | 0.2272 | 0.1020 |
| ma_60_ratio | True | momentum/oscillator | -0.0972 | -0.0018 | -0.0854 | -0.1604 | 0.0862 |
| macd_ratio | True | momentum/oscillator | -0.0867 | -0.0022 | -0.1023 | -0.1510 | 0.0855 |
| rsi | True | momentum/oscillator | -0.0852 | -0.0033 | -0.0769 | -0.1211 | 0.0716 |

`atr_ratio`와 `intraday_range_ratio`는 현재 `MODEL_FEATURE_COLUMNS`에 없으므로 indicator-only 후보로만 분리했다. 특히 ATR 계열은 band 폭과 직접 연결되지만, 이번 CP에서는 feature contract를 바꾸지 않는다.

## 6. 약하거나 노이즈일 수 있는 피처

| 피처 | 그룹 | finite | shift | abs IC | 노이즈 가능성 | leakage 점검 |
| --- | --- | --- | --- | --- | --- | --- |
| open_ratio | price/return | 1.0000 | 0.0096 | -0.0117 | 단일 시점 갭이라 노이즈가 크고 range 피처와 중복될 수 있다. | 낮음 |
| vol_change | volume/liquidity | 1.0000 | 0.0023 | 0.0235 | 꼬리가 두꺼워 clipping이나 log 변환 없이 쓰면 band 폭을 불안정하게 만들 수 있다. | 낮음 |
| bb_position | momentum/oscillator | 1.0000 | 0.0809 | -0.0889 | 정상 범위 밖 값이 있어 약한 winsorization 후보지만 이번 CP에서는 변경하지 않는다. | 낮음 |
| us10y | market/sector/breadth | 1.0000 | 2.7496 | -0.0004 | 동일 날짜 모든 ticker에 같은 값이라 cross-section band 차별력은 약할 수 있다. | 중간 |
| yield_spread | market/sector/breadth | 1.0000 | 2.2273 | -0.0622 | 공통 macro 값이라 개별 ticker band 폭 분화에는 약할 수 있다. | 중간 |
| credit_spread_hy | market/sector/breadth | 1.0000 | 7.4067 | 0.0145 | 초기 0 대체 구간이 있어 regime flag처럼 동작할 수 있다. | 중간 |
| ma200_pct | market/sector/breadth | 1.0000 | 0.1958 | -0.1419 | 공통 breadth 값이라 ticker별 band 차별력은 제한적일 수 있다. | 중간 |
| regime_calm | market/sector/breadth | 1.0000 | 0.5166 | -0.1520 | 원-핫 regime은 세밀한 폭 조정보다 거친 상태 구분에 가깝다. | 낮음 |
| regime_neutral | market/sector/breadth | 1.0000 | 0.7121 | 0.0342 | 단독 정보량은 낮고 다른 macro 피처와 중복될 수 있다. | 낮음 |
| revenue | fundamentals | 1.0000 | 2.3387 | 0.0168 | 원시 규모값과 0 대체 비중이 커서 노이즈 또는 size 편향 위험이 크다. | 중간 |
| net_income | fundamentals | 1.0000 | 22.6559 | 0.0142 | 0 대체와 원시 규모값 때문에 그대로 쓰면 왜곡 가능성이 크다. | 중간 |
| equity | fundamentals | 1.0000 |  | 0.0208 | 원시 규모값, 0 대체, sector size 편향이 크다. | 중간 |
| eps | fundamentals | 1.0000 |  | 0.0132 | 이상치와 0 대체 비중이 커서 band에는 약할 가능성이 높다. | 중간 |
| roe | fundamentals | 1.0000 |  | 0.0146 | 분모 효과 이상치와 0 대체 비중이 커서 노이즈 위험이 있다. | 중간 |
| debt_ratio | fundamentals | 1.0000 |  | 0.0254 | 분모 효과와 이상치가 커서 현재 원시값은 약하거나 불안정할 수 있다. | 중간 |
| has_macro | missingness flag | 1.0000 |  |  | 상수에 가까워 band 정보량이 거의 없다. | 낮음 |
| has_breadth | missingness flag | 1.0000 | 0.3460 | 0.0021 | 거의 상수라 실제 band 폭 조정력은 제한적이다. | 낮음 |
| has_fundamentals | missingness flag | 1.0000 | 2.3824 | 0.0142 | 재무 가용성 자체를 학습해 편향을 만들 위험이 있다. | 중간 |
| day_of_week_sin | calendar/future covariate | 1.0000 | 0.0057 | -0.0117 | 단독 band 예측력은 약할 가능성이 높다. | 낮음 |
| day_of_week_cos | calendar/future covariate | 1.0000 | 0.0275 | -0.0025 | 단독 band 예측력은 약할 가능성이 높다. | 낮음 |
| month_sin | calendar/future covariate | 1.0000 | 0.2911 | 0.0428 | 시장 regime보다 설명력이 약할 가능성이 높다. | 낮음 |
| month_cos | calendar/future covariate | 1.0000 | 0.2746 | 0.0251 | 단독 정보량은 낮다. | 낮음 |
| is_month_end | calendar/future covariate | 1.0000 | 0.0225 | 0.0189 | 효과가 특정 기간에만 나타날 수 있어 불안정하다. | 낮음 |
| is_quarter_end | calendar/future covariate | 1.0000 | 0.0239 | -0.0026 | 희소 binary라 작은 smoke에서는 불안정할 수 있다. | 낮음 |
| is_opex_friday | calendar/future covariate | 1.0000 | 0.0077 | -0.0057 | 희소 이벤트라 단독으로는 약할 가능성이 높다. | 낮음 |

해석상 가장 조심할 그룹은 fundamentals와 missingness flag다. 재무 원시값은 결측 0 대체와 기업 규모 편향이 크고, `has_fundamentals`는 재무 가용성 자체를 학습할 위험이 있다. macro/breadth/regime은 band regime에는 도움이 될 수 있지만 동일 날짜 모든 ticker가 같은 값을 갖기 때문에 종목별 lower/upper 폭 분화에는 약할 수 있다.

## 7. 피처 그룹 실험 matrix

| 세트 | 피처 수 | 계약 | 우선순위 | 설명 |
| --- | --- | --- | --- | --- |
| A. price_return_only | 5 | 현 계약 | 비교용 | 가격/수익률/ratio 중심 최소 세트다. |
| B. price_volatility | 10 | 현 계약 | BM 1순위 | CNN-LSTM band의 1순위 smoke feature_set이다. 현재 계약 안에서는 ATR 대신 high/low/bb_position을 쓴다. |
| C. price_volatility_volume | 11 | 현 계약 | BM 2순위 보조 | 거래량 충격이 band breach proxy를 보강하는지 확인한다. |
| D. technical_only | 11 | 현 계약 | TiDE 비교용 | 종목 자체 기술 패턴만으로 direct/param band가 살아나는지 본다. |
| E. no_fundamentals | 30 | 현 계약 | 노이즈 제거 | 재무 원시값의 0 대체와 size 편향을 제거한다. |
| F. no_market_context | 25 | 현 계약 | ablation | market/common-date context 없이 종목 자체 패턴만 확인한다. |
| G. full_features | 36 | 현 계약 | 기준선 | 현 36개 전체 입력이며 기존 기준선이다. |
| H. candidate_indicator_expanded | 13 | 변경 필요 | Phase 1 이후 후보 | 이번 CP에서는 설계와 proxy만 남기고 실제 모델 feature 추가는 하지 않는다. |

세부 column 목록과 contract 변경 사항은 `docs/cp63_bm_feature_set_plan.json`에 기록했다. `candidate_indicator_expanded`는 설계만 남기며 실제 모델 feature 추가는 이번 CP 범위 밖이다.

## 8. 다음 CP BM smoke 계획

| 실험 | feature_set | epochs | limit_tickers | selector | W&B | 이유 |
| --- | --- | --- | --- | --- | --- | --- |
| CNN-LSTM seq60 q15-b2 direct | price_volatility | 3 | 50 | band_gate | off 권장 | 기존 s60 q15-b2 direct가 dynamic width는 있으나 baseline을 못 이겨, market/fundamental 노이즈 제거 효과를 확인한다. |
| CNN-LSTM seq60 q20-b2 direct | price_volatility | 3 | 50 | band_gate | off 권장 | q15보다 nominal coverage가 낮은 설정에서 width/interval trade-off가 개선되는지 본다. |
| TiDE q10-b2 param | technical_only | 3 | 50 | band_gate | off 권장 | param width 방식은 watch 가능성이 있으나 interval_score가 약해 feature noise 제거 효과를 확인한다. |
| TiDE q10-b2 direct | price_volatility | 3 | 50 | band_gate | off 권장 | 현재 fail에 가까운 direct 구조가 feature pruning으로도 살아나지 못하면 후순위로 낮춘다. |

공통 조건은 `role=band_model`, `band_target_type=raw_future_return`, `line_target_type`은 band 평가 미사용, `checkpoint_selection=band_gate`, `save-run=false`, full run 금지다. smoke에서는 W&B off를 권장하고, 비교 로그가 꼭 필요할 때만 명시적으로 켠다.

## 9. 통계 baseline 위치

Bollinger, rolling quantile, constant width train quantile, volatility scaled constant band는 비교 기준으로만 둔다. Phase 1 BM 본류에서는 AI가 통계 baseline residual 또는 scale 보정값을 예측하는 모델을 구현하지 않는다. baseline-aware 보정은 Phase 1.5 후보로 문서화한다.

## 10. 검증

- 새 학습 없음.
- save-run 없음.
- DB 쓰기 없음.
- DB schema 변경 없음.
- UI/backend API 수정 없음.
- composite/overlay 지표 사용 없음.
- 생성 CSV/JSON은 pandas/json으로 파싱 확인 대상이다.
- 산출물은 UTF-8로 작성했다.
