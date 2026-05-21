# CP118-BM 1W Band Feature/Target 설계 감사 보고서

## 1. 결론

- 상태: PASS
- 새 학습: 없음
- save-run / DB write / inference 저장 / W&B / composite / frontend 수정: 없음
- 데이터 소스: `C:\Users\user\lens\data\parquet`의 yfinance local 1W snapshot만 사용
- feature contract: `v3_adjusted_ohlc`, `MODEL_N_FEATURES=36` 유지
- 다음 실험 feature group: 7개 확정
- target 우선순위: `raw_future_return` quantile band를 1W BM v1 기본으로 유지
- ATR/indicator-only 판단: `atr_ratio`, `intraday_range_ratio`는 proxy가 강하지만 현재 모델 feature가 아니므로 제품 저장 전 즉시 승격하지 않고 별도 feature contract CP 후보로 분리
- candidate registry: 단일 winner 방식이 아니라 `recommended_default`, `selectable_candidate`, `research_watch`로 관리

1W BM v1의 raw 기준 기본 후보는 CP114의 `cnn_h4_q10_pvv_direct`다. 다만 `cnn_h4_q15_pvv_direct`와 `tide_h4_q15_pvv_param`도 버리지 않고 selectable candidate로 남긴다.

## 2. 입력과 로컬 스냅샷 확인

| 항목 | 값 |
|---|---|
| provider | `yfinance` |
| snapshot dir | `C:\Users\user\lens\data\parquet` |
| indicator snapshot | `indicators_yfinance_1W.parquet` |
| price snapshot | `price_data_yfinance_1W.parquet` |
| stock info snapshot | `stock_info.parquet` |
| 분석 rows | 52,900 |
| ticker count | 100 |
| date range | 2016-02-19 ~ 2026-04-03 |
| horizon | 4주 |
| severe downside threshold | raw h4 return 하위 10%, -0.075443 |

torch import나 학습 파이프라인은 사용하지 않았다. `ai/preprocessing.py`와 `backend/app/services/feature_svc.py`의 원문 상수만 파싱해 feature contract를 확인했다.

## 3. 36개 모델 피처 분류

| feature | 그룹 | band 관점 해석 | 약점 가능성 |
|---|---|---|---|
| `log_return` | price/return | 최근 방향성과 단기 변동성의 최소 신호 | 1W h4 폭 예측 proxy는 중간 이하 |
| `open_ratio` | price/return | 주간 gap/시가 왜곡이 downside event와 약하게 연결 | 단독 신호는 약함 |
| `high_ratio` | volatility/range | 주간 상단 range 확장 신호 | 방향성보다 폭 신호로 해석 필요 |
| `low_ratio` | volatility/range | 하단 range 확장과 future volatility proxy가 강함 | 부호가 음수라 모델 해석이 비직관적일 수 있음 |
| `vol_change` | volume/liquidity | 거래량 충격과 유동성 상태 proxy | 이번 1W proxy에서는 상위권 아님 |
| `ma_5_ratio` | price/return | 단기 추세 이격과 range 확대 가능성 | downside magnitude 신호 약함 |
| `ma_20_ratio` | price/return | 중기 이격, realized range/vol proxy가 있음 | 폭 전용 피처로는 ATR보다 약함 |
| `ma_60_ratio` | price/return | 장기 이격, regime성 가격 위치 | 반응이 느릴 수 있음 |
| `rsi` | momentum/oscillator | 과열/침체가 future range와 음의 단조 관계 | 방향성/폭 의미가 섞임 |
| `macd_ratio` | momentum/oscillator | trend momentum과 변동 확대 proxy | 1W proxy 강도는 중간 |
| `bb_position` | volatility/range | band 위치와 mean reversion/range 확대 신호 | 직접 폭 지표는 아니며 위치 신호에 가까움 |
| `us10y` | market/sector/breadth | 금리 regime 후보 | yfinance 1W snapshot에서는 상수 0 |
| `yield_spread` | market/sector/breadth | 금리 곡선 regime 후보 | yfinance 1W snapshot에서는 상수 0 |
| `vix_close` | market/sector/breadth | 시장 변동성 regime 후보 | yfinance 1W snapshot에서는 상수 0 |
| `credit_spread_hy` | market/sector/breadth | credit stress regime 후보 | yfinance 1W snapshot에서는 상수 0 |
| `nh_nl_index` | market/sector/breadth | breadth stress 후보 | yfinance 1W snapshot에서는 상수 0 |
| `ma200_pct` | market/sector/breadth | 시장 breadth 후보 | yfinance 1W snapshot에서는 상수 0 |
| `regime_calm` | market/sector/breadth | market regime one-hot 후보 | 이번 snapshot에서는 상수에 가까움 |
| `regime_neutral` | market/sector/breadth | market regime one-hot 후보 | 이번 snapshot에서는 상수에 가까움 |
| `regime_stress` | market/sector/breadth | market regime one-hot 후보 | 이번 snapshot에서는 상수에 가까움 |
| `revenue` | fundamentals | 기업 규모/성장성 proxy | local 1W band에서는 full과 no_fundamentals 차이가 거의 없음 |
| `net_income` | fundamentals | 이익 체력 proxy | weekly h4 tail에는 느릴 수 있음 |
| `equity` | fundamentals | 재무 안정성 proxy | 1W 단기 band 폭에는 약할 가능성 |
| `eps` | fundamentals | 주당 이익 proxy | 결측/정적 성격으로 noise 가능 |
| `roe` | fundamentals | 수익성 proxy | 1W h4 폭 직접 신호로 약함 |
| `debt_ratio` | fundamentals | 레버리지 risk proxy | 단기 band 폭보다 cross-sectional risk에 가까움 |
| `has_macro` | missingness flag | macro coverage 여부 | 이번 snapshot에서는 분산 없음 |
| `has_breadth` | missingness flag | breadth coverage 여부 | 이번 snapshot에서는 분산 없음 |
| `has_fundamentals` | missingness flag | fundamentals coverage 여부 | 이번 snapshot에서는 분산 없음 |
| `day_of_week_sin` | calendar/future covariate | 주간 anchor calendar 위치 | 1W에서는 의미가 제한적 |
| `day_of_week_cos` | calendar/future covariate | 주간 anchor calendar 위치 | 1W에서는 의미가 제한적 |
| `month_sin` | calendar/future covariate | 계절성/월중 regime proxy | seasonal noise 가능 |
| `month_cos` | calendar/future covariate | 계절성/월중 regime proxy | downside proxy가 있지만 과해석 금지 |
| `is_month_end` | calendar/future covariate | 월말 rebalancing/event proxy | event calendar noise 가능 |
| `is_quarter_end` | calendar/future covariate | 분기말 event proxy | 표본 빈도가 낮음 |
| `is_opex_friday` | calendar/future covariate | 옵션 만기 주간 후보 | 1W 집계에서는 의미가 희석될 수 있음 |

## 4. 1W Band Proxy Feature Audit

Spearman IC의 부호는 단조 방향을 뜻한다. band 폭 예측에서는 절댓값이 더 중요하다.

| feature | in model | abs return IC | range IC | h4 realized vol IC | downside magnitude IC | severe downside IC |
|---|---:|---:|---:|---:|---:|---:|
| `atr_ratio` | false | 0.282389 | 0.575606 | 0.467658 | 0.038833 | 0.122672 |
| `intraday_range_ratio` | false | 0.223180 | 0.463371 | 0.377869 | 0.039478 | 0.110024 |
| `low_ratio` | true | -0.117922 | -0.242347 | -0.203630 | -0.010880 | -0.043427 |
| `bb_position` | true | -0.124981 | -0.214778 | -0.182973 | 0.012544 | -0.017123 |
| `rsi` | true | -0.115934 | -0.191554 | -0.160594 | 0.005071 | -0.022808 |
| `high_ratio` | true | 0.080024 | 0.170694 | 0.130383 | 0.021713 | 0.052785 |
| `ma_20_ratio` | true | -0.089881 | -0.136360 | -0.118031 | 0.018969 | -0.003456 |
| `ma_60_ratio` | true | -0.082699 | -0.106388 | -0.089663 | 0.004399 | -0.019529 |
| `ma_5_ratio` | true | -0.060583 | -0.100473 | -0.087537 | 0.019626 | 0.001283 |
| `macd_ratio` | true | -0.072001 | -0.084954 | -0.068806 | 0.009661 | -0.011749 |

해석:

- 현재 36개 안에서는 `low_ratio`, `bb_position`, `rsi`, `high_ratio`, `ma_20_ratio`가 band 폭 proxy 상위권이다.
- `atr_ratio`와 `intraday_range_ratio`는 realized range, future realized volatility, severe downside event에서 모두 강하다.
- `atr_ratio`는 indicator snapshot에는 있으나 모델 feature에는 없다.
- `intraday_range_ratio`는 이번 감사에서 OHLC로 계산한 indicator-only 후보이며 모델 feature가 아니다.
- market/regime 계열은 local yfinance 1W snapshot에서 대부분 상수 0이라 이번 proxy로는 기여 판단이 불가능하다.
- leakage 의심 feature는 없었다. 다만 `realized_range`, `future_realized_volatility`, `downside_magnitude`는 target 후보이지 입력 feature로 넣으면 안 된다.

## 5. Feature Group 설계

| feature_set | n | 계약 변경 | abs IC 평균 | range IC 평균 | vol IC 평균 | downside IC 평균 | severe IC 평균 | 판단 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `price_return_only` | 5 | 없음 | 0.0557 | 0.0837 | 0.0720 | 0.0145 | 0.0103 | ablation |
| `price_volatility_volume` | 11 | 없음 | 0.0718 | 0.1213 | 0.1010 | 0.0123 | 0.0183 | BM v1 기본 |
| `technical_only` | 11 | 없음 | 0.0718 | 0.1213 | 0.1010 | 0.0123 | 0.0183 | 현재는 PVV와 동일 |
| `market_regime_only` | 11 | 없음 | NA | NA | NA | NA | NA | snapshot 상수, 독립 실험 보류 |
| `pvv_plus_market_regime` | 22 | 없음 | 0.0718 | 0.1213 | 0.1010 | 0.0123 | 0.0183 | 시장 컨텍스트 데이터 확보 후 재평가 |
| `no_fundamentals` | 30 | 없음 | 0.0577 | 0.0947 | 0.0787 | 0.0128 | 0.0203 | watch, val 안정성 필요 |
| `full_features` | 36 | 없음 | 0.0577 | 0.0947 | 0.0787 | 0.0128 | 0.0203 | 기존 기준선 |
| `pvv_plus_atr_candidate` | 13 | 필요 | 0.0996 | 0.1826 | 0.1505 | 0.0165 | 0.0333 | 다음 feature contract CP 후보 |

확정된 다음 실험 그룹:

1. `price_volatility_volume`: 1W BM v1 기본.
2. `price_return_only`: 가격/수익률 최소 피처 ablation.
3. `no_fundamentals`: fundamentals 제거가 val gate를 안정화할 수 있는지 재검증.
4. `full_features`: 기존 36개 기준선.
5. `pvv_plus_market_regime`: macro/breadth가 살아 있는 snapshot에서만 의미 있음.
6. `market_regime_only`: 현재 snapshot에서는 연구 보류.
7. `pvv_plus_atr_candidate`: feature contract 변경 CP에서만 실행.

## 6. Target 후보 설계

| target 후보 | 우선순위 | 장점 | 단점 | 코드 변경 범위 | 제품 해석 |
|---|---:|---|---|---|---|
| `raw_future_return` q_low/q_high | 1 | 제품 lower/upper band와 직접 연결, CP112~CP114 검증 완료 | tail 방향성/폭 정렬이 loss에 충분히 강제되지 않을 수 있음 | 없음 | 가장 명확함 |
| `realized_range` | 3 | 폭 target으로 ATR proxy와 잘 맞음 | 방향성과 lower/upper를 직접 만들지 못함 | 중간 | band width 설명에는 좋지만 band 자체는 후처리 필요 |
| `realized_volatility` | 3 | dynamic width alignment에 적합 | 실제 수익률 interval coverage와 다름 | 중간 | 변동성 표시로는 명확하나 band 해석은 간접 |
| `downside_magnitude` | 2 | lower breach와 risk-first 설계에 직접적 | 0이 많은 sparse target | 중간 | 하방 위험 보조 설명에 좋음 |
| `tail_event_probability` | 3 | severe downside event 감지에 적합 | threshold 의존, calibration 필요 | 중간~큼 | risk probability로 분리 표시 가능 |
| hybrid: quantile band + width auxiliary | 2 | raw band 해석을 유지하면서 width 정렬 개선 가능 | loss balancing과 checkpoint 기준 설계 필요 | 큼 | 제품 band는 raw quantile로 유지 가능 |

결론: 1W BM v1은 `raw_future_return` quantile band를 유지한다. target 변경은 제품 후보 저장 전 본류가 아니라 별도 설계 CP로 둔다. 다만 `downside_magnitude` 또는 `realized_volatility` auxiliary는 다음 연구 후보로 가치가 있다.

## 7. Direct / Param 설계 감사

| 방식 | CP114 연결 | 장점 | 약점 | 다음 방향 |
|---|---|---|---|---|
| direct lower/upper | `cnn_h4_q10_pvv_direct`, `cnn_h4_q15_pvv_direct` | 제품 산출물인 lower/upper와 직접 연결, q10 후보가 coverage/interval/dynamic width를 동시 개선 | q 범위에 따라 과보수 가능, lower/upper crossing이나 비대칭 관리 필요 | CNN-LSTM direct를 1W BM v1 기본으로 유지 |
| param center+width | `tide_h4_q15_pvv_param` | 폭이 좁고 `band_width_ic=0.326849`로 가장 강함 | upper breach 0.185739, downside_width_ic 0.005363으로 하방 대응 약함 | selectable candidate로 유지, center는 제품 line으로 사용 금지 |

band_model의 line 또는 center 출력은 제품 예측선이 아니다. 제품에서 쓰는 산출물은 lower/upper band series다.

## 8. Loss 조정 후보

이번 CP에서는 구현하지 않는다.

| loss 후보 | 개선하려는 지표 | 기대 효과 | 주의점 |
|---|---|---|---|
| lower breach penalty 강화 | `lower_breach_rate`, `asymmetric_interval_score` | risk-first 하방 이탈 억제 | 폭이 과도하게 넓어질 수 있음 |
| width sharpness penalty | `avg_band_width`, `p90_band_width`, `interval_score` | coverage만 높고 너무 넓은 band 방지 | 너무 강하면 coverage 붕괴 |
| dynamic width alignment loss | `band_width_ic`, width bucket ratio | 폭이 실제 변동성과 같이 움직이도록 유도 | realized vol target 계산과 horizon 정합 필요 |
| downside width alignment loss | `downside_width_ic`, lower breach | 하방 위험 구간에서 하방 폭 확대 | sparse downside target으로 불안정 가능 |
| squeeze/regime auxiliary loss | `squeeze_breakout_rate`, regime별 coverage | 좁은 band 이후 breakout 실패 감소 | regime label 품질이 먼저 필요 |

## 9. Calibration 분리 원칙

- raw band로 살아나는 후보: `cnn_h4_q10_pvv_direct`, `cnn_h4_q15_pvv_direct`, `tide_h4_q15_pvv_param`.
- calibration 전제 후보: raw coverage가 애매하지만 dynamic width가 강한 param/auxiliary 후보.
- calibration 결과를 제품 성능으로 과대해석하지 않는다.
- 1W BM v1은 가능하면 raw 기준 생존 후보를 우선한다.
- rolling quantile/Bollinger/constant width baseline은 비교 기준이며 residual/scale 보정 모델은 이번 본류가 아니다.

## 10. Regime별 평가 설계

다음 CP에서 같은 raw prediction을 regime별로 slice해 평가한다.

| regime split | 정의 후보 | 봐야 할 metric |
|---|---|---|
| high volatility / low volatility | realized h4 vol 또는 `atr_ratio` 분위수 | `coverage_abs_error`, `lower_breach_rate`, `band_width_ic`, `p90_band_width` |
| rising market / falling market | trailing 13주 시장 또는 ticker return | `lower_breach_rate`, `upper_breach_rate`, `asymmetric_interval_score` |
| wide band / narrow band | 예측 band width 분위수 | `coverage_abs_error`, `squeeze_breakout_rate`, `p90_band_width` |
| high ATR / low ATR | `atr_ratio` 분위수, contract 변경 전에는 audit-only | `band_width_ic`, `downside_width_ic`, `squeeze_breakout_rate` |
| sector known / unknown | `stock_info.parquet` sector 존재 여부 | `coverage_abs_error`, `lower_breach_rate`, `interval_score` |

## 11. Candidate Registry

| category | display_name | source_experiment_id | 핵심 강점 | 약점 | best use |
|---|---|---|---|---|---|
| recommended_default | 1W CNN-LSTM q10/q90 PVV direct | `cnn_h4_q10_pvv_direct` | coverage_abs_error 0.006014, interval 0.301977, band_width_ic 0.282361 | val overcoverage seed 재현성 필요 | 기본 1W band 후보 |
| selectable_candidate | 1W CNN-LSTM q15/q85 PVV direct | `cnn_h4_q15_pvv_direct` | raw band_gate 통과, q10보다 덜 넓은 정책 대안 | q10 대비 coverage/interval/dynamic width 열세 | q10 과보수 의심 시 |
| selectable_candidate | 1W TiDE q15/q85 PVV param | `tide_h4_q15_pvv_param` | 가장 좁은 폭, band_width_ic 0.326849 | upper breach 높고 downside_width_ic 약함 | 좁은 band와 dynamic width 우선 |
| research_watch | 1W CNN-LSTM q15/q85 no fundamentals | `cnn_h4_q15_no_fundamentals_direct` | test interval/width 매력적 | val band_gate fail | feature pruning 연구 |
| research_watch | 1W CNN-LSTM H8 feasibility | `cnn_h8_q15_pvv_direct_feasibility` | h8 실행 가능성 확인 | h6~h8 downside_width_ic 음수, interval 악화 | 장기 horizon 연구 |

프론트는 추후 GPT 5.5 / 5.4 선택처럼 band model 선택 UI로 확장 가능하게 문서화한다. 단, 지금은 UI/backend 수정 범위가 아니다.

## 12. 다음 CP 추천

1. 1W BM raw save-run 후보 CP: `cnn_h4_q10_pvv_direct` seed 재현성 확인 후 제품 저장 후보로 진행.
2. 1W regime evaluation CP: CP114 후보들을 high/low vol, rising/falling, wide/narrow band로 분리 평가.
3. indicator feature contract CP: `atr_ratio`, `intraday_range_ratio`를 모델 feature로 승격할지 별도 계약 변경안 작성.
4. target/loss 연구 CP: raw quantile band 유지 + width/downside auxiliary loss 설계.

## 13. 산출물

- `docs/cp118_bm_1w_band_feature_target_audit_report.md`
- `docs/cp118_bm_1w_band_feature_target_audit_metrics.json`
- `docs/cp118_bm_1w_feature_proxy_metrics.csv`
- `docs/cp118_bm_1w_feature_group_plan.json`

