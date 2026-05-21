# CP55 후보별 공정 baseline 비교 및 band 후보군 재확장 보고서

CP55는 새 학습이 아니라, 기존 checkpoint와 기존 산출물을 후보별 동일 조건의 통계 baseline과 비교하는 CP다.

## Executive Summary
- 이번 baseline은 모델 baseline이 아니라 통계 baseline이다. DLinear/NLinear/LightGBM/CatBoost 같은 학습형 baseline은 아직 아니다.
- line baseline(minimal): zero_line, momentum_line_horizon, reversal_line_horizon, random_or_shuffled_score_seed_0
- band baseline(minimal): constant_width_train_quantile, rolling_historical_quantile_band_w60, rolling_bollinger_return_band_w60_k1
- line 후보 2개, band 후보 6개를 정리했다.
- 실행 모드: limited, baseline_set=minimal, 후보별 timeout=300.0초, 총 실행시간=540.04초.
- CP52 전체 band 지표가 없는 과거 후보는 새 추론을 돌리지 않고 `재실험 필요` 또는 부분 비교로 표시했다.
- 기존 45분 지연 원인은 후보 전체와 rolling baseline 전체를 한 번에 계산했고, 동일 split baseline 캐시와 후보 범위 제한이 부족했기 때문이다. 이번 스크립트는 discover/smoke/제한 실행으로 분리했다.

## 실행 시간 및 병목
- cp45_188_confirm_metrics::s60_q15_b2_direct_188::scalar_width: elapsed=197.11s, baseline=197.10s, limit=200
- cp45_actual_metrics::s60_q15_b2_direct::scalar_width: elapsed=132.88s, baseline=132.86s, limit=100
- h5_longer_context_seq252_p32_s16: elapsed=85.08s, baseline=85.06s, limit=50
- tide_param_scalar_width: elapsed=66.75s, baseline=66.74s, limit=50
- cp45_cnn_lstm_band_sweep_metrics::s60_q20_b2_direct::scalar_width: elapsed=38.87s, baseline=38.86s, limit=100

## Line 후보 공정 비교
| AI 후보 | family | AI IC | best baseline | baseline IC | AI spread | false_safe_tail | severe_recall | 판정 |
|---|---|---:|---|---:|---:|---:|---:|---|
| h5_longer_context_seq252_p32_s16 | patchtst | 0.0241 | historical_mean_line_w60 | 0.0211 | 0.0034 | 0.1475 | 0.8514 | 후보 |
| h5_dense_overlap_seq252_p16_s4 | patchtst | -0.0049 | historical_mean_line_w60 | 0.0211 | -0.0018 | 0.0079 | 0.9910 | 탈락 |

## Band 후보 공정 비교
| AI 후보 | family | baseline | AI interval | baseline interval | AI cov err | baseline cov err | AI width_ic | baseline width_ic | AI 승 | baseline 승 | 판정 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cp45_188_confirm_metrics::s60_q15_b2_direct_188::scalar_width | cnn_lstm | rolling_bollinger_return_band_w60_k1 | - | 0.1307 | 0.1081 | 0.0015 | - | 0.3283 | 0 | 1 | 재실험 필요 |
| cp45_actual_metrics::s60_q15_b2_direct::scalar_width | cnn_lstm | rolling_bollinger_return_band_w60_k1 | - | 0.1282 | 0.0986 | 0.0006 | - | 0.3289 | 0 | 1 | 재실험 필요 |
| cp45_cnn_lstm_band_sweep_metrics::s60_q20_b2_direct::scalar_width | cnn_lstm | rolling_bollinger_return_band_w60_k1 | - | 0.1115 | 0.1934 | 0.0994 | - | 0.3289 | 0 | 1 | 재실험 필요 |
| patchtst_band_reference::h5_longer_context_seq252_p32_s16 | patchtst | rolling_bollinger_return_band_w60_k1 | 0.6649 | 0.1055 | 0.4905 | 0.2000 | 0.1886 | 0.3395 | 0 | 4 | 탈락 |
| tide_direct_original | tide | rolling_historical_quantile_band_w60 | 1.0960 | 0.1598 | 0.2860 | 0.0500 | 0.0758 | 0.3566 | 0 | 4 | 탈락 |
| tide_param_scalar_width | tide | rolling_historical_quantile_band_w60 | 0.8268 | 0.1598 | 0.1685 | 0.0500 | 0.1809 | 0.3566 | 0 | 4 | 탈락 |

## Band 판정 요약
- 상태별: {"재실험 필요": 3, "탈락": 3}
- family별: {"cnn_lstm": {"재실험 필요": 3}, "tide": {"탈락": 2}, "patchtst": {"탈락": 1}}

## 핵심 해석
- PatchTST는 line 후보로 유지한다. band head는 과보수/통계 baseline 대비 약한 후보가 많아 line 전용 해석이 맞다.
- CNN-LSTM은 일부 dynamic width 신호가 있지만, 후보별 fair baseline에서 rolling quantile/Bollinger를 안정적으로 이겼다고 보기 어렵다.
- TiDE는 checkpoint는 남아 있으나 CP52 전체 지표가 부족한 산출물이 많다. 기존 regrade 기준으로도 downside_width_ic가 약해 주력 band 후보는 아니다.
- 통계 baseline이 강하므로 다음 band CP는 AI가 직접 lower/upper를 예측하는 방식보다 baseline-aware residual/scale band를 우선 검토해야 한다.

## 재실험 또는 미지원 목록
- cp45_188_confirm_metrics::s60_q15_b2_direct_188::original (band): 기본 후보 범위 제한 max_candidates=8
- s60_q15_b2_direct_188 (band): 기본 후보 범위 제한 max_candidates=8
- cp45_actual_metrics::s60_q15_b2_direct::original (band): 기본 후보 범위 제한 max_candidates=8
- s60_q15_b2_direct (band): 기본 후보 범위 제한 max_candidates=8
- h5_baseline_seq252_p16_s8 (line): 기본 후보 범위 제한 max_candidates=8
- cp37_role_based_model_recheck_metrics::A_patchtst_line_gate (line): 기본 후보 범위 제한 max_candidates=8
- cp45_cnn_lstm_band_sweep_metrics::s60_q20_b2_direct::original (band): 기본 후보 범위 제한 max_candidates=8
- h10_baseline_seq252_p16_s8 (line): 기본 후보 범위 제한 max_candidates=8
- h10_dense_overlap_seq252_p16_s4 (line): 기본 후보 범위 제한 max_candidates=8
- h10_longer_context_seq252_p32_s16 (line): 기본 후보 범위 제한 max_candidates=8
- h20_baseline_seq252_p16_s8 (line): 기본 후보 범위 제한 max_candidates=8
- h20_baseline_seq504_p16_s8 (line): 기본 후보 범위 제한 max_candidates=8
- h20_dense_overlap_seq252_p16_s4 (line): 기본 후보 범위 제한 max_candidates=8
- h20_longer_context_seq252_p32_s16 (line): 기본 후보 범위 제한 max_candidates=8
- s60_q20_b2_direct (band): 기본 후보 범위 제한 max_candidates=8
- cp32_patchtst_clean_feature_revalidation_metrics::50_q15_b2 (band): 기본 후보 범위 제한 max_candidates=8
- cp32_patchtst_clean_feature_revalidation_metrics::50_q20_b2 (band): 기본 후보 범위 제한 max_candidates=8
- cp32_patchtst_clean_feature_revalidation_metrics::50_q25_b2 (band): 기본 후보 범위 제한 max_candidates=8
- cp32_patchtst_clean_feature_revalidation_metrics::50_baseline_q10_b1 (band): 기본 후보 범위 제한 max_candidates=8
- cp33_patchtst_revin_ablation_metrics::A_q25_b2_revin_on (band): 기본 후보 범위 제한 max_candidates=8
- cp33_patchtst_revin_ablation_metrics::B_q25_b2_revin_off (band): 기본 후보 범위 제한 max_candidates=8
- cp33_patchtst_revin_ablation_metrics::C_q30_b2_revin_off (band): 기본 후보 범위 제한 max_candidates=8
- cp33_patchtst_revin_ablation_metrics::D_q35_b2_revin_off (band): 기본 후보 범위 제한 max_candidates=8
- cp35_tide_cnn_band_rescue_smoke_metrics::A_tide_q15_b2 (band): 기본 후보 범위 제한 max_candidates=8
- cp35_tide_cnn_band_rescue_smoke_metrics::B_tide_q10_b2 (band): 기본 후보 범위 제한 max_candidates=8
- cp35_tide_cnn_band_rescue_smoke_metrics::C_cnn_lstm_seq120_q20_b2 (band): 기본 후보 범위 제한 max_candidates=8
- cp35_tide_cnn_band_rescue_smoke_metrics::D_cnn_lstm_seq60_q20_b2 (band): 기본 후보 범위 제한 max_candidates=8
- cp35_tide_cnn_band_rescue_smoke_metrics::E_tide_q10_b2_param (band): 기본 후보 범위 제한 max_candidates=8
- cp37_role_based_model_recheck_metrics::B_tide_band_gate_param (band): 기본 후보 범위 제한 max_candidates=8
- cp37_role_based_model_recheck_metrics::C_tide_band_gate_direct (band): 기본 후보 범위 제한 max_candidates=8
- cp37_role_based_model_recheck_metrics::D_cnn_lstm_band_gate_seq120 (band): 기본 후보 범위 제한 max_candidates=8
- cp37_role_based_model_recheck_metrics::E_cnn_lstm_band_gate_seq60 (band): 기본 후보 범위 제한 max_candidates=8
- cp38::cnn_lstm_seq60::original (band): 기본 후보 범위 제한 max_candidates=8
- cp38::cnn_lstm_seq60::scalar_width (band): 기본 후보 범위 제한 max_candidates=8
- cp38::tide_param::original (band): 기본 후보 범위 제한 max_candidates=8
- cp45_actual_metrics::s60_q10_b2_direct::original (band): 기본 후보 범위 제한 max_candidates=8
- cp45_actual_metrics::s60_q10_b2_direct::scalar_width (band): 기본 후보 범위 제한 max_candidates=8
- cp45_actual_metrics::s90_q15_b2_direct::original (band): 기본 후보 범위 제한 max_candidates=8
- cp45_actual_metrics::s90_q15_b2_direct::scalar_width (band): 기본 후보 범위 제한 max_candidates=8
- cp45_cnn_lstm_band_sweep_metrics::s45_q20_b2_direct::original (band): 기본 후보 범위 제한 max_candidates=8

## 다음 단계 제안
- A안: rolling historical quantile 또는 Bollinger return band 위에 residual scale을 학습하는 baseline-aware band로 전환한다.
- B안: TiDE branch는 CP52 full regrade를 먼저 한 뒤 dynamic width가 살아나는 경우에만 재개한다.
- C안: CNN-LSTM은 direct lower/upper sweep보다 baseline 대비 residual correction으로 좁힌다.
- D안: PatchTST는 line 전용으로 유지하고 band 후보 경쟁에서는 분리한다.
