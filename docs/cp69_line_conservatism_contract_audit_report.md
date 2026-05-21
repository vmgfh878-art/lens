# CP69-LM line 보수성 계약 재감사

## 1. 범위
- 새 학습, 성능 재평가, h20 추가 보정, DB 쓰기, save-run, band/composite/UI/backend 작업은 하지 않았다.
- checkpoint는 `torch.load(..., map_location='cpu')`로 config만 읽었다.

## 2. line loss 구현 감사
| 항목 | 값 | 판정 |
| --- | --- | --- |
| line loss class | AsymmetricHuberLoss | PASS |
| composite line loss | AsymmetricHuberLoss | PASS |
| error definition | prediction - target | PASS |
| overprediction condition | error > 0 | PASS |
| alpha/beta/delta | 1.0/2.0/1.0 | PASS |
| unit over/under loss | 1.0000/0.5000 | PASS |

단위 예제에서 target=1.0, prediction=2.0은 overprediction이며 loss=1.0이다. prediction=0.0은 underprediction이며 loss=0.5다. 같은 절대오차에서 beta=2가 overprediction에 적용된다.

## 3. checkpoint/config 감사 매트릭스
| 후보 | source | 판정 | alpha | beta | delta | lambda_line | lambda_band | lambda_cross | selection | target | band_target | h | seq | patch | feature_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h5_longer_context_seq252_p32_s16 | CP49 | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 5 | 252 | 32/16 | full_features |
| h5_baseline_seq252_p16_s8 | CP49 | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 5 | 252 | 16/8 | full_features |
| h5_dense_overlap_seq252_p16_s4 | CP49 | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 5 | 252 | 16/4 | full_features |
| h20_longer_context_seq252_p32_s16 | CP65_reference | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 20 | 252 | 32/16 | full_features |
| h20_technical_only_seq252_p32_s16 | CP65 | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 20 | 252 | 32/16 | technical_only |
| h20_no_fundamentals_seq252_p32_s16 | CP65 | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 20 | 252 | 32/16 | no_fundamentals |
| h20_price_volatility_volume_seq252_p32_s16 | CP65 | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 20 | 252 | 32/16 | technical_only / source:price_volatility_volume |
| h20_full_features_post_backfill_100 | CP67 | verified_trained_conservative | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 20 | 252 | 32/16 | full_features |
| cp68_horizon_bucket_downshift | CP68 | posthoc_only_not_training | 1.0000 | 2.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | line_gate | raw_future_return | raw_future_return | 20 | 252 | 32/16 | full_features |

## 4. 분류 요약
| 분류 | 개수 |
| --- | --- |
| verified_trained_conservative | 8 |
| posthoc_only_not_training | 1 |

## 5. h5 제품 후보 신뢰도
- `h5_longer_context_seq252_p32_s16`, `h5_baseline_seq252_p16_s8`, `h5_dense_overlap_seq252_p16_s4` 모두 checkpoint config에서 alpha=1, beta=2, delta=1, lambda_line=1, line_target_type=raw_future_return, checkpoint_selection=line_gate를 확인했다.
- 따라서 h5 후보들은 `verified_trained_conservative`로 분류한다.
- 단 CP49 계열 checkpoint는 config에 `feature_set` 값이 저장되지 않아 source metrics 기준 full_features로 추정한다. 이 문제는 보수 loss 확인과는 별개다.
- CP65 `price_volatility_volume` 후보는 당시 `technical_only`와 feature 정의가 동일해 같은 checkpoint를 재사용했다. 그래서 checkpoint config에는 `technical_only`로 남아 있고 source feature_set은 `price_volatility_volume`이다.

## 6. CP68 용어 재분류
- `raw_model_line`: 모델 checkpoint가 직접 출력한 line이다.
- `trained_conservative_line`: alpha/beta asymmetric Huber loss로 학습된 raw_model_line이다.
- `display_calibrated_line`: validation offset으로 아래로 보정한 표시선이다.
- CP68 `global_downshift`와 `horizon_bucket_downshift`는 학습 보수성이 아니라 post-hoc display calibration이다.
- 따라서 CP68 best line은 `posthoc_only_not_training`으로 분류하고, underlying CP67 checkpoint만 `verified_trained_conservative`로 둔다.

## 7. 결론
주요 h5/h20 PatchTST checkpoint는 line loss 계약상 하방 보수 학습이 확인됐다. CP68의 보수선은 학습으로 생긴 보수성이 아니라 표시 보정이므로 제품/문서에서 반드시 `display_calibrated_line`으로 구분해야 한다.

## 8. 검증
- `.venv\Scripts\python.exe -m py_compile ai\cp69_line_conservatism_contract_audit.py ai\tests\test_losses.py`: 통과
- `python -m json.tool docs\cp69_line_conservatism_contract_audit_matrix.json`: 통과
- `.venv\Scripts\python.exe -m unittest ai.tests.test_losses ai.tests.test_loss`: 8개 통과
