# CP54 기준선 지표 비교 보고서

CP54는 새 학습이 아니라 CP52/CP53 표준 지표판을 단순 baseline에도 적용해 AI 후보가 실제 기준선을 이겼는지 확인하는 CP다.

## Executive Summary
- 평가 공간: raw_future_return, timeframe=1D, horizon=5, limit_tickers=50.
- 기준 quantile: q_low=0.15, q_high=0.85, nominal_coverage=0.7.
- Line baseline은 zero, historical mean, momentum, reversal, shuffled score를 계산했다.
- Band baseline은 constant train quantile, rolling historical quantile, return-space Bollinger, volatility-scaled constant band를 계산했다.
- AI 후보 수: line 3개, band 4개, composite policy 3개.
- 결론: PatchTST line은 단순 line 기준선을 대체로 이겼지만, CNN-LSTM band는 rolling historical quantile/Bollinger baseline을 interval_score와 coverage_abs_error에서 아직 못 이겼다.

## 판정 질문 답변
- patchtst_line_vs_simple_baselines: PatchTST가 line 기준선보다 우세합니다. PatchTST h5 longer-context test IC=0.024122550822042437, spread=0.003399287596530438. 최고 IC baseline은 random_or_shuffled_score IC=0.021256706809453896. false_safe_tail은 PatchTST=0.14751046998604, 최저 baseline reversal_line_horizon=0.4859818520241973입니다.
- cnn_lstm_band_vs_baselines: CNN-LSTM band가 baseline을 이기지 못했습니다. CNN-LSTM band coverage_abs_error=0.10693387985229497, interval_score=0.14870473742485046. 최저 interval baseline은 rolling_historical_quantile_band_w252=0.13104186952114105, 최저 coverage error baseline은 rolling_bollinger_return_band_w60_k1=3.4880638122514185e-05입니다.
- dynamic_width_vs_volatility_baseline: CNN-LSTM의 동적 폭 신호는 양수지만, 최고 단순 baseline보다 약합니다. CNN-LSTM band_width_ic=0.2462876143335152, downside_width_ic=0.03331726296496214. 최고 width baseline은 rolling_historical_quantile_band_w252 band_width_ic=0.37800740585018927입니다.
- wide_band_or_dynamic_risk: 단순히 넓어서 맞추는 후보로만 보기는 어렵지만, baseline 대비 우위는 아직 부족합니다. nominal 대비 오차는 생존권이고 폭-위험 상관은 양수입니다.
- h5_h20_branch_policy: h5와 h20은 계속 분리합니다. CP53 기준 h20 후보는 별도 branch 성격이며, CP54 baseline도 h5 raw-return 비교판으로 제한했습니다.

## Line Baseline
| 후보 | IC | spread | IC_IR | false_safe_tail | severe_recall | downside_capture | sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero_line | - | -0.0003 | - | 1.0000 | 0.0000 | 1.0000 | -0.0103 |
| historical_mean_line_w20 | -0.0044 | -0.0014 | -0.0178 | 0.5330 | 0.4746 | 0.2450 | -0.0447 |
| historical_mean_line_w60 | 0.0211 | 0.0031 | 0.0761 | 0.5541 | 0.4495 | 0.2392 | 0.0617 |
| momentum_line_horizon | -0.0120 | -0.0046 | -0.0509 | 0.5150 | 0.4906 | 0.2369 | -0.1673 |
| reversal_line_horizon | 0.0120 | 0.0046 | 0.0509 | 0.4860 | 0.5085 | 0.2371 | 0.0818 |
| random_or_shuffled_score | 0.0213 | 0.0019 | 0.1346 | 0.5113 | 0.4989 | 0.2205 | -0.0578 |

## AI Line 후보
| 후보 | IC | spread | IC_IR | false_safe_tail | severe_recall | downside_capture | sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| h5_baseline_seq252_p16_s8 | 0.0136 | 0.0022 | 0.0559 | 0.2214 | 0.7746 | 0.2330 | 0.0380 |
| h5_longer_context_seq252_p32_s16 | 0.0241 | 0.0034 | 0.1082 | 0.1475 | 0.8514 | 0.2439 | 0.0191 |
| h5_dense_overlap_seq252_p16_s4 | -0.0049 | -0.0018 | -0.0186 | 0.0079 | 0.9910 | 0.2549 | -0.0664 |

## Band Baseline
| 후보 | nominal | empirical | abs_error | width | interval | width_ic | downside_ic | squeeze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_width_train_quantile | 0.7000 | 0.6591 | 0.0409 | 0.0508 | 0.1406 | 0.2682 | 0.0658 | 0.0835 |
| rolling_historical_quantile_band_w60 | 0.7000 | 0.6541 | 0.0459 | 0.0570 | 0.1361 | 0.3587 | 0.0745 | 0.0602 |
| rolling_historical_quantile_band_w120 | 0.7000 | 0.6673 | 0.0327 | 0.0569 | 0.1332 | 0.3667 | 0.0825 | 0.0569 |
| rolling_historical_quantile_band_w252 | 0.7000 | 0.6737 | 0.0263 | 0.0567 | 0.1310 | 0.3780 | 0.0929 | 0.0550 |
| rolling_bollinger_return_band_w20_k1 | 0.7000 | 0.6242 | 0.0758 | 0.0584 | 0.1471 | 0.3112 | 0.0650 | 0.0820 |
| rolling_bollinger_return_band_w20_k1.5 | 0.7000 | 0.7866 | 0.0866 | 0.0875 | 0.1392 | 0.3112 | 0.0650 | 0.0820 |
| rolling_bollinger_return_band_w20_k2 | 0.7000 | 0.8772 | 0.1772 | 0.1167 | 0.1482 | 0.3112 | 0.0650 | 0.0820 |
| rolling_bollinger_return_band_w60_k1 | 0.7000 | 0.7000 | 0.0000 | 0.0634 | 0.1336 | 0.3395 | 0.0641 | 0.0683 |
| rolling_bollinger_return_band_w60_k1.5 | 0.7000 | 0.8491 | 0.1491 | 0.0951 | 0.1330 | 0.3395 | 0.0641 | 0.0683 |
| rolling_bollinger_return_band_w60_k2 | 0.7000 | 0.9224 | 0.2224 | 0.1267 | 0.1485 | 0.3395 | 0.0641 | 0.0683 |
| volatility_scaled_constant_band_w20 | 0.7000 | 0.5766 | 0.1234 | 0.0473 | 0.1430 | 0.3406 | 0.0627 | 0.0676 |
| volatility_scaled_constant_band_w60 | 0.7000 | 0.6043 | 0.0957 | 0.0492 | 0.1397 | 0.3490 | 0.0639 | 0.0606 |

## AI Band 후보
| 후보 | nominal | empirical | abs_error | width | interval | width_ic | downside_ic | squeeze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s60_q15_b2_direct | 0.7000 | 0.7978 | 0.0978 | 0.1073 | 0.1596 | 0.2502 | 0.0321 | 0.0864 |
| s60_q15_b2_direct_188 | 0.7000 | 0.8069 | 0.1069 | 0.1021 | 0.1487 | 0.2463 | 0.0333 | 0.0713 |
| tide_param_scalar_width | 0.8000 | 0.6315 | 0.1685 | 0.1149 | 0.8268 | 0.1809 | -0.0051 | 0.1975 |
| tide_direct_original | 0.8000 | 0.5140 | 0.2860 | 0.1312 | 1.0960 | 0.0758 | -0.0025 | 0.2299 |

## Composite 정책 참고
| 정책 | line_inside | warning | conservative_false_safe | width_increase | coverage | upper_breach |
|---|---:|---:|---:|---:|---:|---:|
| raw_composite | 0.0000 | 0.7298 | 0.0000 | 1.0000 | 0.7649 | 0.2160 |
| risk_first_lower_preserve | 1.0000 | 0.0000 | 0.0000 | 1.8270 | 0.8064 | 0.1915 |
| risk_first_upper_buffer_1.10 | 1.0000 | 0.0000 | 0.0000 | 1.9848 | 0.8957 | 0.1021 |

## 기존 판정 변화
- CP53의 PatchTST h5 longer-context 후보는 line 기준선과 나란히 비교한다. baseline이 일부 지표에서 더 좋으면 AI가 아직 못 이긴 영역으로 기록한다.
- CNN-LSTM band 후보는 nominal 대비 coverage_abs_error와 interval_score를 baseline과 직접 비교한다.
- Composite는 모델 성능 판정이 아니라 제품 표시/정책 지표로 유지한다.

## 남은 리스크
- CP54는 기존 CP53 AI 산출물을 재사용하고 baseline만 새로 계산했다. AI checkpoint 재추론을 새로 돌리지 않았다.
- h20 branch는 h5 baseline 표와 직접 경쟁시키지 않았다.
- random_or_shuffled_score는 후보가 아니라 downside_capture_rate 무작위 기준 확인용이다.

## 다음 CP 추천
- AI line이 momentum/reversal보다 약한 지표는 feature 또는 objective 재점검 대상으로 둔다.
- AI band가 Bollinger/rolling quantile보다 interval_score에서 밀리면 CNN-LSTM band sweep 대신 baseline-aware calibration을 먼저 해야 한다.
- h5와 h20은 계속 branch를 분리해서 보고한다.
