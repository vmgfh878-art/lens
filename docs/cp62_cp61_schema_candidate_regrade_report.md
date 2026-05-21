# CP62 CP61 schema 기준 기존 후보 재채점 보고서

CP62는 새 학습 없이 기존 checkpoint / metrics JSON 산출물을 CP61의 line_metrics / band_metrics 분리 schema로 다시 해석한 CP다.

## 1. Executive Summary
- 새 학습, save-run, DB 쓰기, DB schema 변경, UI/API 수정은 수행하지 않았다.
- composite/overlay 계열은 모델 랭킹에서 제외했고, legacy_overlay_diagnostics로만 분리했다.
- line 후보 집계: {'survive': 5, 'watch': 4, 'fail': 1}.
- band 후보 집계: {'survive': 0, 'watch': 4, 'fail': 2}.
- top line 후보: h5_longer_context_seq252_p32_s16, h20_longer_context_seq252_p32_s16, h20_dense_overlap_seq252_p16_s4.
- top band 후보: s60_q15_b2_direct, s60_q15_b2_direct_188, tide_param_scalar_width.

## 2. CP61 schema 기준 후보 목록
- line 후보는 `line_metrics`만으로 판정했다. coverage/breach/line_inside_band는 line 판정에 쓰지 않았다.
- band 후보는 `band_metrics`만으로 판정했다. IC/spread/line_inside_band는 band 판정에 쓰지 않았다.
- composite 정책 후보는 `line_band_composite`, `risk_first_*`, `include_line_clamp`, `upper_buffer` 전부 랭킹 제외다.

## 3. Line 후보 순위
| 후보 | horizon | 구조 | IC | spread | IC_IR | false_safe_tail | severe_recall | baseline | AI 승/패 | 판정 |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| h5_baseline_seq252_p16_s8 | 5 | patch_len=16, patch_stride=8 | 0.0136 | 0.0022 | 0.0559 | 0.2214 | 0.7746 | historical_mean_line_w60 | 3/2 | line_survive |
| h5_longer_context_seq252_p32_s16 | 5 | patch_len=32, patch_stride=16 | 0.0241 | 0.0034 | 0.1082 | 0.1475 | 0.8514 | historical_mean_line_w60 | 5/0 | line_survive |
| h5_dense_overlap_seq252_p16_s4 | 5 | patch_len=16, patch_stride=4 | -0.0049 | -0.0018 | -0.0186 | 0.0079 | 0.9910 | historical_mean_line_w60 | 3/2 | risk_only_watch |
| h10_baseline_seq252_p16_s8 | 10 | patch_len=16, patch_stride=8 | 0.0088 | 0.0010 | 0.0341 | 0.1880 | 0.8127 | historical_mean_line_w60 | 3/2 | line_survive |
| h10_longer_context_seq252_p32_s16 | 10 | patch_len=32, patch_stride=16 | -0.0074 | -0.0038 | -0.0289 | 0.2725 | 0.7241 | historical_mean_line_w60 | 3/2 | line_fail |
| h10_dense_overlap_seq252_p16_s4 | 10 | patch_len=16, patch_stride=4 | 0.0138 | -0.0003 | 0.0585 | 0.0839 | 0.9172 | historical_mean_line_w60 | 3/2 | line_watch |
| h20_baseline_seq252_p16_s8 | 20 | patch_len=16, patch_stride=8 | -0.0068 | 0.0022 | -0.0338 | 0.2676 | 0.7321 | historical_mean_line_w60 | 3/2 | line_watch |
| h20_longer_context_seq252_p32_s16 | 20 | patch_len=32, patch_stride=16 | 0.0199 | 0.0072 | 0.0820 | 0.3149 | 0.6868 | historical_mean_line_w60 | 4/1 | line_survive |
| h20_dense_overlap_seq252_p16_s4 | 20 | patch_len=16, patch_stride=4 | 0.0138 | 0.0018 | 0.0653 | 0.1091 | 0.8899 | historical_mean_line_w60 | 3/2 | line_survive |
| h20_baseline_seq504_p16_s8 | 20 | patch_len=16, patch_stride=8 | -0.0521 | -0.0264 | -0.2176 | 0.0388 | 0.9604 | random_or_shuffled_score | 3/2 | risk_only_watch |

## 4. Band 후보 순위
| 후보 | 모델 | q | coverage err | lower/upper breach | width | interval | width_ic | downside_ic | baseline | AI 승/패 | 판정 |
|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---|
| s60_q20_b2_direct | cnn_lstm | 0.20~0.80 | 0.1947 | 0.1163/0.0890 | 0.1043 | 0.1422 | 0.2839 | 0.0479 | rolling_bollinger_return_band_w60_k1 | 0/4 | band_watch |
| s60_q15_b2_direct | cnn_lstm | 0.15~0.85 | 0.0978 | 0.1187/0.0835 | 0.1073 | 0.1596 | 0.2502 | 0.0321 | rolling_bollinger_return_band_w60_k1 | 0/4 | band_watch |
| s60_q15_b2_direct_188 | cnn_lstm | 0.15~0.85 | 0.1069 | 0.0844/0.1087 | 0.1021 | 0.1487 | 0.2463 | 0.0333 | rolling_bollinger_return_band_w60_k1 | 0/4 | band_watch |
| tide_param_scalar_width | tide | 0.10~0.90 | 0.1685 | 0.2148/0.1538 | 0.1149 | 0.8268 | 0.1809 | -0.0051 | rolling_historical_quantile_band_w60 | 0/4 | band_watch |
| tide_direct_original | tide | 0.10~0.90 | 0.2860 | 0.3119/0.1742 | 0.1312 | 1.0960 | 0.0758 | -0.0025 | rolling_historical_quantile_band_w60 | 0/4 | band_fail |
| patchtst_band_reference::h5_longer_context_seq252_p32_s16 | patchtst | 0.25~0.75 | 0.4905 | 0.0042/0.0053 | 0.6627 | 0.6649 | 0.1886 | 0.0216 | rolling_bollinger_return_band_w60_k1 | 0/4 | band_fail |

## 5. Baseline 대비 이긴 점 / 진 점
### Line Baseline
| baseline | IC | spread | false_safe_tail | severe_recall |
|---|---:|---:|---:|---:|
| zero_line | - | -0.0003 | 1.0000 | 0.0000 |
| historical_mean_line_w20 | -0.0044 | -0.0014 | 0.5330 | 0.4746 |
| historical_mean_line_w60 | 0.0211 | 0.0031 | 0.5541 | 0.4495 |
| momentum_line_horizon | -0.0120 | -0.0046 | 0.5150 | 0.4906 |
| reversal_line_horizon | 0.0120 | 0.0046 | 0.4860 | 0.5085 |
| random_or_shuffled_score | 0.0213 | 0.0019 | 0.5113 | 0.4989 |

### Band Baseline
| baseline | coverage err | interval | width_ic | downside_ic |
|---|---:|---:|---:|---:|
| constant_width_train_quantile | 0.0409 | 0.1406 | 0.2682 | 0.0658 |
| rolling_historical_quantile_band_w60 | 0.0459 | 0.1361 | 0.3587 | 0.0745 |
| rolling_historical_quantile_band_w120 | 0.0327 | 0.1332 | 0.3667 | 0.0825 |
| rolling_historical_quantile_band_w252 | 0.0263 | 0.1310 | 0.3780 | 0.0929 |
| rolling_bollinger_return_band_w20_k1 | 0.0758 | 0.1471 | 0.3112 | 0.0650 |
| rolling_bollinger_return_band_w20_k1.5 | 0.0866 | 0.1392 | 0.3112 | 0.0650 |
| rolling_bollinger_return_band_w20_k2 | 0.1772 | 0.1482 | 0.3112 | 0.0650 |
| rolling_bollinger_return_band_w60_k1 | 0.0000 | 0.1336 | 0.3395 | 0.0641 |
| rolling_bollinger_return_band_w60_k1.5 | 0.1491 | 0.1330 | 0.3395 | 0.0641 |
| rolling_bollinger_return_band_w60_k2 | 0.2224 | 0.1485 | 0.3395 | 0.0641 |
| volatility_scaled_constant_band_w20 | 0.1234 | 0.1430 | 0.3406 | 0.0627 |
| volatility_scaled_constant_band_w60 | 0.0957 | 0.1397 | 0.3490 | 0.0639 |

요약:
- line 후보는 IC/spread가 양수인 PatchTST h5/h20 계열이 남지만, random/historical baseline과의 격차가 크지 않은 후보가 있다.
- band 후보는 dynamic width 신호는 일부 있으나, rolling/Bollinger/constant 통계 baseline 대비 interval과 coverage_abs_error를 안정적으로 이겼다고 보기 어렵다.

## 6. Composite/Overlay 지표 제외 확인
- `line_inside_band_ratio`, `risk_first_lower_preserve`, `risk_first_upper_buffer`, `include_line_clamp`는 모델 생존 근거에서 제외했다.
- 기존 composite 산출물은 legacy demo 또는 표시 진단 자료로만 분류했다.
- CP62 JSON의 `legacy_overlay_diagnostics.excluded_from_model_ranking=true`로 기록했다.

## 7. 재실험 필요 후보
- 재실험 필요 후보 수: 58.
- 주요 band watch 후보: s60_q20_b2_direct, s60_q15_b2_direct, s60_q15_b2_direct_188, tide_param_scalar_width.
- line survive 후보: h5_baseline_seq252_p16_s8, h5_longer_context_seq252_p32_s16, h10_baseline_seq252_p16_s8, h20_longer_context_seq252_p32_s16, h20_dense_overlap_seq252_p16_s4.
- band survive 후보: 없음.

## 8. 기존 판정에서 바뀐 후보
- CP61 기준에서는 composite compatibility로 band 후보를 살리지 않는다.
- CNN-LSTM s60 계열은 단독 band 지표상 watch/survive 가능성이 있어도, 통계 baseline을 압도하지 못하면 재실험 필요로 분리했다.
- PatchTST dense overlap h5는 risk-only 참고 가능성이 있으나 line 주력 후보로는 약하다.

## 9. 다음 CP 추천
- line 실험: PatchTST h5 longer/baseline을 기준으로 TiDE line 후보가 있으면 같은 line_metrics로 재평가한다.
- band 실험: CNN-LSTM/TiDE/PatchTST band를 통계 baseline-aware residual/scale 방식으로 재설계할지 검토한다.
- 데이터/피처 점검: band_width_ic와 downside_width_ic가 baseline보다 약한 원인이 feature/target 계약인지 확인한다.
- baseline 강화: rolling historical quantile, Bollinger return band, volatility scaled band를 제품 baseline으로도 유지할 가치가 있다.
