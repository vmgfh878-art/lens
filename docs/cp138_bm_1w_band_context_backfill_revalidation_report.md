# CP138-BM: 1W Band Context Backfill 이후 후보 재검증

## 1. 결론
- 기본 후보: `없음 - raw validation 제품 기준 통과 후보 없음`
- selectable_verified: `없음`
- rejected: `cnn_pvv_q10_direct, cnn_pvv_q15_direct, cnn_no_fundamentals_q10_direct, cnn_full_q10_direct, cnn_full_q10_direct_lower_guard_w1p5`
- design_needed: `context_light_q10_direct`
- 후보 선택은 validation과 raw band 기준으로만 수행했다. test metric은 read-only 기록으로만 남겼다.

## 2. 데이터와 금지 조건
- provider/source: `yfinance` / `yfinance`
- source_data_hash: `13a7f83d`
- context_column_checksum: `0751103c151849d2`
- CP133 1W indicator_value_checksum: `b83cb84767f8c357`
- price parquet: `C:\Users\user\lens\data\parquet\price_data_yfinance_1W.parquet`, mtime `2026-05-04T13:15:45`
- indicator parquet: `C:\Users\user\lens\data\parquet\indicators_yfinance_1W.parquet`, mtime `2026-05-06T15:14:01`
- feature_version: `v3_adjusted_ohlc`
- MODEL_FEATURE_COLUMNS: `36`
- atr_ratio 모델 feature 포함: `False`
- feature NaN/Inf total: `0`
- target NaN/Inf total: `0`
- save-run/DB write/inference 저장/W&B/composite/live fetch/EODHD 호출은 수행하지 않았다.

## 3. Feature Set 확인
| feature_set | exists | columns | atr | range | error |
| --- | --- | --- | --- | --- | --- |
| context_light | False |  |  |  | 알 수 없는 feature_set입니다: context_light. 허용값: candidate_indicator_expanded, full_features, no_fundamentals, no_market_context, price_return_only, price_volatility, price_volatility_volume, technical_only |
| full_features | True | 36 | False | False |  |
| no_fundamentals | True | 30 | False | False |  |
| price_volatility_volume | True | 11 | False | False |  |

## 4. 후보 결과표
| candidate | category | model | feature_set | q_low | gate | cov_abs | lower | upper | interval | p90_w | bw_ic | down_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_pvv_q10_direct | rejected | cnn_lstm | price_volatility_volume | 0.100000 | 0.500000 | 0.085571 | 0.041739 | 0.072689 | 0.265048 | 0.328967 | 0.265318 | 0.018944 |
| cnn_pvv_q15_direct | rejected | cnn_lstm | price_volatility_volume | 0.150000 | 0.500000 | 0.085580 | 0.091189 | 0.123231 | 0.255678 | 0.295435 | 0.221239 | 0.013771 |
| cnn_no_fundamentals_q10_direct | rejected | cnn_lstm | no_fundamentals | 0.100000 | 0.000000 | 0.246903 | 0.256815 | 0.190088 | 0.425335 | 0.226813 | 0.130913 | 0.010327 |
| cnn_full_q10_direct | rejected | cnn_lstm | full_features | 0.100000 | 0.000000 | 0.196580 | 0.159772 | 0.236808 | 0.391352 | 0.286582 | 0.146237 | 0.001657 |
| cnn_full_q10_direct_lower_guard_w1p5 | rejected | cnn_lstm | full_features | 0.100000 | 0.000000 | 0.221654 | 0.195898 | 0.225756 | 0.408954 | 0.272237 | 0.138304 | 0.003185 |
| tide_pvv_q15_param | experiment_record | tide | price_volatility_volume | 0.150000 | 1.000000 | 0.001686 | 0.120173 | 0.178141 | 0.243948 | 0.175292 | 0.319981 | 0.017220 |
| context_light_q10_direct | design_needed | cnn_lstm | context_light | 0.100000 | 0.000000 |  |  |  |  |  |  |  |

## 5. Regime 하방 이탈
| candidate | falling lower | high vol lower | high ATR lower | failures |
| --- | --- | --- | --- | --- |
| cnn_pvv_q10_direct | 0.091693 | 0.056691 | 0.047738 | band_gate_pass,coverage_abs_error |
| cnn_pvv_q15_direct | 0.181400 | 0.105084 | 0.093204 | band_gate_pass,coverage_abs_error |
| cnn_no_fundamentals_q10_direct | 0.443032 | 0.245108 | 0.248515 | band_gate_pass,coverage_abs_error,lower_breach_rate,falling_regime_lower_breach_rate,high_volatility_lower_breach_rate,high_atr_lower_breach_rate,band_width_ic,asymmetric_interval_score_competitive |
| cnn_full_q10_direct | 0.297492 | 0.166798 | 0.162168 | band_gate_pass,coverage_abs_error,falling_regime_lower_breach_rate,band_width_ic,asymmetric_interval_score_competitive |
| cnn_full_q10_direct_lower_guard_w1p5 | 0.353098 | 0.197327 | 0.193353 | band_gate_pass,coverage_abs_error,lower_breach_rate,falling_regime_lower_breach_rate,band_width_ic,asymmetric_interval_score_competitive |
| tide_pvv_q15_param | 0.241432 | 0.153171 | 0.153389 | falling_regime_lower_breach_rate |
| context_light_q10_direct |  |  |  | feature_set 미구현 |

## 6. 해석
- `full_features`가 우위로 남으면 CP133 context backfill이 macro/breadth/fundamentals 정보를 실제 후보 판정에 반영한 것으로 해석한다.
- `price_volatility_volume`이 우위면 해석 가능한 최소 피처 우선 원칙에 따라 PVV를 기본 후보로 둔다.
- `no_fundamentals`가 우위면 fundamentals의 희소성 또는 0 대체 노이즈를 기록하고 full 기본화를 보류한다.
- `tide_pvv_q15_param`은 falling/high-vol/high-ATR lower breach가 유지되는지로 대안 후보 여부를 판단했다.
- `context_light_q10_direct`는 현재 `docs/cp63_bm_feature_set_plan.json` 계약에 없으면 이번 CP에서 새로 구현하지 않고 design_needed로 남겼다.

## 7. 산출물
- metrics: `docs\cp138_bm_1w_band_context_backfill_revalidation_metrics.json`
- registry: `docs\cp138_bm_1w_band_candidate_registry.json`
- summary csv: `docs\cp138_bm_1w_band_revalidation_summary.csv`
- logs: `docs\cp138_bm_1w_band_context_backfill_revalidation_logs`
