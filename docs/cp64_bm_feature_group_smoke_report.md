# CP64-BM CNN-LSTM/TiDE band feature group smoke

## 1. 실행 계약

이번 CP는 AI band layer 전용 smoke다. line 실험, composite/overlay 지표, 통계 baseline residual/scale 보정은 수행하지 않았다. `ai.train`에는 `--feature-set`과 alias `--feature-columns-preset`을 추가했고, CP63의 `docs/cp63_bm_feature_set_plan.json`에서 feature set을 로드한다. `full_features`는 기존 36개 기본 동작을 유지한다.

feature set 적용 경로는 다음과 같다.

1. CLI에서 `--feature-set`을 수신한다.
2. CP63 JSON에서 columns를 로드한다.
3. `MODEL_FEATURE_COLUMNS`의 부분집합인지 검증한다.
4. indicator-only 또는 contract 변경 필요 set은 차단한다.
5. train/val/test feature tensor와 mean/std를 같은 순서로 축소한다.
6. `TrainConfig.n_features`를 모델 생성에 반영한다.

공통 조건은 `feature_version=v3_adjusted_ohlc`, `role=band_model`, `band_target_type=raw_future_return`, `checkpoint_selection=band_gate`, `limit_tickers=50`, `epochs=3`, `batch_size=256`, `save-run=false`, `wandb=off`다.

## 2. feature_set별 실제 column 수

| feature_set | n_features | columns |
| --- | --- | --- |
| price_volatility | 10 | log_return, open_ratio, high_ratio, low_ratio, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position |
| price_volatility_volume | 11 | log_return, open_ratio, high_ratio, low_ratio, vol_change, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position |
| no_fundamentals | 30 | log_return, open_ratio, high_ratio, low_ratio, vol_change, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position, us10y, yield_spread, vix_close, credit_spread_hy, nh_nl_index, ma200_pct, regime_calm, regime_neutral, regime_stress, has_macro, has_breadth, has_fundamentals, day_of_week_sin, day_of_week_cos, month_sin, month_cos, is_month_end, is_quarter_end, is_opex_friday |
| technical_only | 11 | log_return, open_ratio, high_ratio, low_ratio, vol_change, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position |

## 3. 실행 상태

기존 완료 결과가 있는 실험은 재실행하지 않고 재사용했다. PASS는 해당 실험 학습 명령이 exit code 0으로 끝났다는 뜻이며, SKIPPED가 있으면 이미 완료된 결과를 재사용했다는 뜻이다.

| 실험 | 상태 | exit | 초 | epoch 평균 | epoch 초 | VRAM MB | 재사용 | 로그 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_s60_q15_b2_direct_price_volatility | PASS | 0 | 198.9 | 45.35 | 48.5,45.0,42.6 | 0.0 | existing_result_reused | docs\cp64_bm_feature_group_smoke_logs\cnn_s60_q15_b2_direct_price_volatility.log |
| cnn_s60_q20_b2_direct_price_volatility | PASS | 0 | 186.8 | 43.10 | 42.7,43.7,42.9 | 0.0 | existing_result_reused | docs\cp64_bm_feature_group_smoke_logs\cnn_s60_q20_b2_direct_price_volatility.log |
| cnn_s60_q15_b2_direct_price_volatility_volume | PASS | 0 | 257.3 | 63.67 | 47.4,77.1,66.5 | 0.0 | existing_result_reused | docs\cp64_bm_feature_group_smoke_logs\cnn_s60_q15_b2_direct_price_volatility_volume.log |
| cnn_s60_q15_b2_direct_no_fundamentals | PASS | 0 | 195.5 | 45.59 | 47.6,43.6,45.6 | 0.0 | existing_result_reused | docs\cp64_bm_feature_group_smoke_logs\cnn_s60_q15_b2_direct_no_fundamentals.log |
| tide_q10_b2_param_technical_only | PASS | 0 | 113.2 | 20.22 | 20.8,20.0,19.9 | 0.0 | existing_result_reused | docs\cp64_bm_feature_group_smoke_logs\tide_q10_b2_param_technical_only.log |
| tide_q10_b2_direct_price_volatility | PASS | 0 | 119.9 | 20.16 | 20.6,20.1,19.8 | 0.0 | existing_result_reused | docs\cp64_bm_feature_group_smoke_logs\tide_q10_b2_direct_price_volatility.log |

## 4. band_metrics: coverage와 breach

| 실험 | 모델 | feature_set | n | nominal | empirical | cov err | lower breach | upper breach | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_s60_q15_b2_direct_price_volatility | cnn_lstm | price_volatility | 10 | 0.7000 | 0.7482 | 0.0482 | 0.1599 | 0.0919 | band_watch |
| cnn_s60_q20_b2_direct_price_volatility | cnn_lstm | price_volatility | 10 | 0.6000 | 0.6228 | 0.0228 | 0.2248 | 0.1524 | band_survive |
| cnn_s60_q15_b2_direct_price_volatility_volume | cnn_lstm | price_volatility_volume | 11 | 0.7000 | 0.6940 | 0.0060 | 0.1515 | 0.1544 | band_survive |
| cnn_s60_q15_b2_direct_no_fundamentals | cnn_lstm | no_fundamentals | 30 | 0.7000 | 0.6380 | 0.0620 | 0.1384 | 0.2236 | band_watch |
| tide_q10_b2_param_technical_only | tide | technical_only | 11 | 0.8000 | 0.7893 | 0.0107 | 0.1478 | 0.0629 | band_survive |
| tide_q10_b2_direct_price_volatility | tide | price_volatility | 10 | 0.8000 | 0.8353 | 0.0353 | 0.0790 | 0.0857 | band_watch |

## 5. band_metrics: width와 dynamic signal

| 실험 | avg width | median width | p90 width | interval | lower penalty | upper penalty | width IC | downside IC | vol ratio | downside ratio | squeeze |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_s60_q15_b2_direct_price_volatility | 0.0932 | 0.0830 | 0.1582 | 0.1596 | 0.0510 | 0.0153 | 0.1549 | 0.0225 | 1.4833 | 0.9844 | 0.1259 |
| cnn_s60_q20_b2_direct_price_volatility | 0.0521 | 0.0474 | 0.0844 | 0.1229 | 0.0537 | 0.0171 | 0.2658 | 0.0305 | 2.0027 | 0.9641 | 0.0909 |
| cnn_s60_q15_b2_direct_price_volatility_volume | 0.0630 | 0.0552 | 0.1072 | 0.1334 | 0.0484 | 0.0219 | 0.3095 | 0.0135 | 2.2566 | 0.9725 | 0.0783 |
| cnn_s60_q15_b2_direct_no_fundamentals | 0.0770 | 0.0737 | 0.1284 | 0.2303 | 0.0712 | 0.0822 | 0.1795 | 0.0100 | 1.6067 | 0.9752 | 0.1591 |
| tide_q10_b2_param_technical_only | 0.0858 | 0.0827 | 0.1071 | 0.1870 | 0.0830 | 0.0182 | 0.3161 | 0.0262 | 2.1998 | 0.9605 | 0.0776 |
| tide_q10_b2_direct_price_volatility | 0.1113 | 0.1058 | 0.1635 | 0.1773 | 0.0445 | 0.0215 | 0.2966 | 0.0355 | 2.0797 | 0.9677 | 0.0932 |

## 6. full_features 대비

delta는 CP64 feature set smoke에서 기존 full_features 후보 값을 뺀 값이다. `coverage_abs_error`와 `asymmetric_interval_score`는 낮을수록 좋다.

| 실험 | full 기준 | 개선 | cov err delta | interval delta | full cov err | full interval |
| --- | --- | --- | --- | --- | --- | --- |
| cnn_s60_q15_b2_direct_price_volatility | s60_q15_b2_direct | YES | -0.0497 | 0.0000 | 0.0978 | 0.1596 |
| cnn_s60_q20_b2_direct_price_volatility | s60_q20_b2_direct | YES | -0.1718 | -0.0193 | 0.1947 | 0.1422 |
| cnn_s60_q15_b2_direct_price_volatility_volume | s60_q15_b2_direct | YES | -0.0919 | -0.0262 | 0.0978 | 0.1596 |
| cnn_s60_q15_b2_direct_no_fundamentals | s60_q15_b2_direct | YES | -0.0358 | 0.0708 | 0.0978 | 0.1596 |
| tide_q10_b2_param_technical_only | tide_param_scalar_width | YES | -0.1578 | -0.6398 | 0.1685 | 0.8268 |
| tide_q10_b2_direct_price_volatility | tide_direct_original | YES | -0.2507 | -0.9187 | 0.2860 | 1.0960 |

## 7. 통계 baseline 대비

비교 baseline은 CP62 산출물의 `rolling_bollinger_return_band_w60_k1`, `rolling_historical_quantile_band_w252`, `constant_width_train_quantile`만 사용했다. 통계 baseline은 비교 기준이며 보정 모델로 쓰지 않았다.

| baseline | cov err | interval | width IC | downside IC |
| --- | --- | --- | --- | --- |
| constant_width_train_quantile | 0.0409 | 0.1406 | 0.2682 | 0.0658 |
| rolling_historical_quantile_band_w252 | 0.0263 | 0.1310 | 0.3780 | 0.0929 |
| rolling_bollinger_return_band_w60_k1 | 0.0000 | 0.1336 | 0.3395 | 0.0641 |

실험별 baseline 승패는 다음과 같다. WIN은 지정 baseline 중 하나 이상에서 `coverage_abs_error` 또는 `asymmetric_interval_score`가 개선됐다는 뜻이다.

| 실험 | 승패 | 이긴 baseline | 최선 cov delta | 최선 interval delta |
| --- | --- | --- | --- | --- |
| cnn_s60_q15_b2_direct_price_volatility | LOSE |  | 0.0072 | 0.0190 |
| cnn_s60_q20_b2_direct_price_volatility | WIN | constant_width_train_quantile, rolling_historical_quantile_band_w252, rolling_bollinger_return_band_w60_k1 | -0.0181 | -0.0177 |
| cnn_s60_q15_b2_direct_price_volatility_volume | WIN | constant_width_train_quantile, rolling_historical_quantile_band_w252, rolling_bollinger_return_band_w60_k1 | -0.0349 | -0.0072 |
| cnn_s60_q15_b2_direct_no_fundamentals | LOSE |  | 0.0211 | 0.0897 |
| tide_q10_b2_param_technical_only | WIN | constant_width_train_quantile, rolling_historical_quantile_band_w252 | -0.0302 | 0.0463 |
| tide_q10_b2_direct_price_volatility | WIN | constant_width_train_quantile | -0.0056 | 0.0367 |

판정 집계는 `{'band_watch': 3, 'band_survive': 3}`다.

## 8. BM 우선순위 재판정

CNN-LSTM 최선 smoke는 `cnn_s60_q20_b2_direct_price_volatility`이고, TiDE 최선 smoke는 `tide_q10_b2_direct_price_volatility`다. 이번 smoke 기준으로도 BM 후보 우선순위는 CNN-LSTM 1순위, TiDE 2순위를 유지한다. 다만 survive 판정은 smoke 조건에서의 생존권이며 제품 band 확정이 아니다. PatchTST band는 이번 CP에서 참고 후순위로 유지한다.

## 9. ATR/daily range feature 승격 여부

`atr_ratio`와 `intraday_range_ratio`는 이번 smoke에서 모델 feature로 추가하지 않았다. CP63 proxy상 band에 유리할 가능성은 있으므로 승격 검토 가치는 있다. 다만 승격하려면 feature contract, cache digest, feature_version, checkpoint 호환성 분리를 함께 바꿔야 하므로 CP64-BM에서는 승격 금지로 닫고, 별도 feature contract CP에서 다룬다.

## 10. 다음 CP 추천

- CNN-LSTM은 `price_volatility`와 `price_volatility_volume`을 중심으로 q/lambda를 좁게 재탐색한다.
- `no_fundamentals`는 dynamic width가 양수지만 interval_score가 약하므로 fundamentals 제거는 보조 가설로만 둔다.
- TiDE는 `technical_only` param을 watch/survive 경계 후보로 두고, direct는 구조 개선 없이는 우선순위를 낮춘다.
- baseline-aware 보정은 Phase 1.5 후보로만 기록한다.
- 다음 순서는 CP64-D indicator full backfill이다.
