# CP65-LM PatchTST line feature set smoke + h20 visual branch

## 1. 실행 계약 확인

이번 CP는 PatchTST `role=line_model`만 다뤘고 제품 산출물 기준은 `line_series`다. 평가는 `line_metrics`만 사용했으며 band coverage, composite/overlay 지표, lower/upper breach로 line 후보를 탈락시키지 않았다.

기존 1D cache만 읽었다. 사용 cache는 `ai\cache\feature_index_1D_1a967362529f_0c1d7f52.pt`와 `ai\cache\features_1D_97ece65766b1_0c1d7f52.pt`이며, DB 백필 데이터 읽기, DB 쓰기, save-run, full 473티커 실행, cache fingerprint 생성/삭제, UI/backend 수정은 하지 않았다. ticker registry는 `ai\cache\ticker_id_map_1d_c3a8729f2b24.json`가 이미 존재하는지 확인만 했다.

공통 조건은 `feature_version=v3_adjusted_ohlc`, `target=raw_future_return`, `line_target_type=raw_future_return`, `checkpoint_selection=line_gate`, `ci_aggregate=target`, `limit_tickers=50`, `epochs=3`, `batch_size=256`, `wandb=off`, `no-compile`, `save-run=false`다.

## 2. 실행 상태

| 실험 | 상태 | 초 | epoch 평균 | 재사용 | 로그 |
| --- | --- | --- | --- | --- | --- |
| h5_technical_only_seq252_p32_s16 | completed | 73.7 | 22.58 |  | docs\cp65_lm_feature_h20_smoke_logs\h5_technical_only_seq252_p32_s16.log |
| h5_no_fundamentals_seq252_p32_s16 | completed | 136.9 | 43.51 |  | docs\cp65_lm_feature_h20_smoke_logs\h5_no_fundamentals_seq252_p32_s16.log |
| h5_price_volatility_volume_seq252_p32_s16 | reused_same_columns | 73.7 | 22.58 | h5_technical_only_seq252_p32_s16 | docs\cp65_lm_feature_h20_smoke_logs\h5_technical_only_seq252_p32_s16.log |
| h20_technical_only_seq252_p32_s16 | completed | 88.7 | 26.45 |  | docs\cp65_lm_feature_h20_smoke_logs\h20_technical_only_seq252_p32_s16.log |
| h20_no_fundamentals_seq252_p32_s16 | completed | 150.8 | 46.95 |  | docs\cp65_lm_feature_h20_smoke_logs\h20_no_fundamentals_seq252_p32_s16.log |
| h20_price_volatility_volume_seq252_p32_s16 | reused_same_columns | 88.7 | 26.45 | h20_technical_only_seq252_p32_s16 | docs\cp65_lm_feature_h20_smoke_logs\h20_technical_only_seq252_p32_s16.log |

`technical_only`와 `price_volatility_volume`은 현재 CP63 feature set 정의상 같은 11개 column이다. 그래서 같은 horizon 안에서는 중복 학습을 피하고 선행 결과를 재사용했다.

## 3. 기존 full_features 기준

| 기준 후보 | h | IC | spread | false_safe_tail | false_safe_severe | severe_recall | bias | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h5_longer_context_seq252_p32_s16 | 5 | 0.0241 | 0.0034 | 0.1475 | 0.1486 | 0.8514 | -0.1093 | line_survive |
| h20_longer_context_seq252_p32_s16 | 20 | 0.0199 | 0.0072 | 0.3149 | 0.3132 | 0.6868 | -0.0336 | line_survive |

## 4. line statistical baseline

| baseline | IC | spread | false_safe_tail | severe_recall |
| --- | --- | --- | --- | --- |
| historical_mean_line_w60 | 0.0211 | 0.0031 | 0.5541 | 0.4495 |
| reversal_line_horizon | 0.0120 | 0.0046 | 0.4860 | 0.5085 |
| random_or_shuffled_score | 0.0213 | 0.0019 | 0.5113 | 0.4989 |

## 5. h5 product line 결과

| 실험 | feature_set | n | IC | spread | dir | false_tail | false_severe | severe_recall | bias | MAE | SMAPE | IC Δfull | tail Δfull | recall Δfull | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h5_technical_only_seq252_p32_s16 | technical_only | 11 | 0.0177 | -0.0003 | 0.4949 | 0.2298 | 0.2320 | 0.7680 | -0.0510 | 0.0735 | 1.5598 | -0.0065 | 0.0822 | -0.0834 | line_watch |
| h5_no_fundamentals_seq252_p32_s16 | no_fundamentals | 30 | 0.0035 | 0.0005 | 0.4902 | 0.2516 | 0.2522 | 0.7478 | -0.0416 | 0.0644 | 1.5369 | -0.0207 | 0.1041 | -0.1036 | line_survive |
| h5_price_volatility_volume_seq252_p32_s16 | price_volatility_volume | 11 | 0.0177 | -0.0003 | 0.4949 | 0.2298 | 0.2320 | 0.7680 | -0.0510 | 0.0735 | 1.5598 | -0.0065 | 0.0822 | -0.0834 | line_watch |

## 6. h20 visual/risk branch 결과

| 실험 | feature_set | n | IC | spread | dir | false_tail | false_severe | severe_recall | bias | MAE | SMAPE | IC Δfull | tail Δfull | recall Δfull | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h20_technical_only_seq252_p32_s16 | technical_only | 11 | 0.0137 | 0.0062 | 0.4752 | 0.3375 | 0.3355 | 0.6645 | -0.0292 | 0.0714 | 1.4790 | -0.0062 | 0.0226 | -0.0222 | phase_1_5_visual_risk_watch |
| h20_no_fundamentals_seq252_p32_s16 | no_fundamentals | 30 | 0.0230 | 0.0147 | 0.4873 | 0.3015 | 0.3007 | 0.6993 | -0.0354 | 0.0739 | 1.4832 | 0.0031 | -0.0135 | 0.0126 | visual_risk_survive |
| h20_price_volatility_volume_seq252_p32_s16 | price_volatility_volume | 11 | 0.0137 | 0.0062 | 0.4752 | 0.3375 | 0.3355 | 0.6645 | -0.0292 | 0.0714 | 1.4790 | -0.0062 | 0.0226 | -0.0222 | phase_1_5_visual_risk_watch |

## 7. feature_set별 개선 여부

| branch | feature_set | full 대비 risk 개선 | 판정 |
| --- | --- | --- | --- |
| h5_product_line | technical_only | NO | line_watch |
| h5_product_line | no_fundamentals | NO | line_survive |
| h5_product_line | price_volatility_volume | NO | line_watch |
| h20_visual_risk | technical_only | NO | phase_1_5_visual_risk_watch |
| h20_visual_risk | no_fundamentals | YES | visual_risk_survive |
| h20_visual_risk | price_volatility_volume | NO | phase_1_5_visual_risk_watch |

## 8. 후보 판단

h5 주력 후보는 `기존 h5_longer_context_seq252_p32_s16 유지`다. feature pruning smoke가 기존 h5 full_features보다 line 종합 우위를 안정적으로 보였을 때만 교체한다.

h20 판단은 `생존`이다. h20은 제품 본류 확정 후보가 아니라 Phase 1.5 visual/risk branch로 분리 기록한다. 제품에는 h5 line을 기본선으로 두고, h20은 중기 위험 방향 보조선 또는 별도 horizon 토글로 표시해야 한다. 같은 차트에서 h5와 h20을 한 후보군처럼 랭킹하거나 평균 내면 안 되고, label에는 horizon과 branch provenance를 명시해야 한다.

## 9. 다음 LM 추천

- h5는 `h5_longer_context_seq252_p32_s16`을 주력 후보로 유지하고, pruned 후보가 IC/spread와 false-safe를 동시에 이길 때만 승격한다.
- h20은 Phase 1.5 visual/risk branch로 남기고, 최소 생존 조건을 만족한 feature_set만 제품 시각성 후보로 둔다.
- `technical_only`와 `price_volatility_volume` 정의가 현재 동일하므로 다음 LM에서는 둘 중 하나를 정리하거나, volume 포함 여부를 실제로 갈라서 재정의한 뒤 비교한다.
- `no_market_context`는 이번 6개 제한 밖이라 보류한다. 추가 지시가 있으면 같은 cache-only 경로로 h5/h20에 붙이면 된다.

## 10. 검증

- `python -m py_compile ai\cp65_lm_feature_h20_smoke.py ai\train.py`: 통과
- `python -m unittest ai.tests.test_feature_set_selection`: 5개 통과
- `python -m json.tool docs\cp65_lm_feature_h20_smoke_metrics.json`: 통과
- 마지막 확인 기준 남은 `python/pythonw` 학습 프로세스 없음
