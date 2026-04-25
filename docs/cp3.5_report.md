# CP3.5 완료

## 1. 출력 정합성 fix
- P1-1 (line head 보존): `Y`
- P1-2 (단일 후처리 함수): [`apply_band_postprocess`](/C:/Users/user/lens/ai/postprocess.py) `Y`
  - val/test/inference 모두 호출 위치 일치: `Y`
- P2-3 (width loss 음수 방지): `Y`

## 2. 출력 헤드 두 방식 (A-3)
- `band_mode="direct"` 구현: `Y`
- `band_mode="param"` 구현 (center + log_half_width): `Y`
- 두 방식 forward shape 동일성 검증: `Y`

## 3. PatchTST (B-2)
체크리스트
- [x] RevIN normalize/denormalize
- [x] Channel Independence (`[B*C, L]` 경로)
- [x] Patching (`patch_len`, `stride`)
- [x] Transformer encoder
- [x] Learned positional embedding
- [x] Line head + Band head 분리
- [x] Init 적용 (`trunc_normal std=0.02`)
- [x] Dropout 위치: input dropout + encoder 내부 + output dropout

- 모델 forward smoke (B=4, L=252, C=29) 통과: `Y`
- 메모리 / GPU / batch size 변경 사항: `없음`

## 4. CNN-LSTM (C-2)
체크리스트
- [x] Conv 블록 (2층 `Conv1d`)
- [x] LayerNorm (Conv 뒤)
- [x] Residual connection (`1x1 Conv` projection)
- [x] LSTM (`n_layers=2`)
- [x] LSTM 출력 LayerNorm
- [x] Attention pooling
- [x] Init 적용
- [x] Dropout 위치: conv 뒤 + attention pooling 뒤 + LSTM 내부

- 모델 forward smoke 통과: `Y`

## 5. TiDE (D-1)
체크리스트
- [x] Feature projection (per-timestep)
- [x] Dense Encoder (`ResidualBlock` 반복)
- [x] Dense Decoder (`ResidualBlock` 반복)
- [x] Temporal Decoder
- [x] Lookback skip
- [x] Init 적용
- [x] Dropout 위치: `ResidualBlock` 내부

생략한 요소
- 미래 covariate 처리: `생략`
- 사유: Lens 입력에는 미래 covariate가 없어서 이번 구현 범위에서 제외

- 모델 forward smoke 통과: `Y`

## 6. Init 적용 (E-1)
- 적용 모델: PatchTST / CNN-LSTM / TiDE 모두 `Y`
- 검증한 `nn.Linear` weight std 평균: `0.02 근처` (`15%` 이상, `25%` 이하 범위 통과)

## 7. Dropout 위치 (E-2)
- PatchTST: input + output 위치 추가 `Y`
- CNN-LSTM: conv 뒤 + attention pooling 뒤 추가 `Y`
- TiDE: `ResidualBlock` 내부 통합 `Y`

## 8. 테스트
- 기존 25 테스트 green: `Y`
- 신규 테스트: `15건`
  - `test_postprocess_preserves_line_head`
  - `test_postprocess_band_sorted`
  - `test_train_val_inference_postprocess_identical`
  - `test_width_loss_nonnegative`
  - `test_band_mode_param_lower_le_upper`
  - `test_patchtst_forward_shape`
  - `test_patchtst_revin_normalization`
  - `test_patchtst_channel_independent_shape`
  - `test_cnn_lstm_forward_shape`
  - `test_cnn_lstm_attention_pooling_sums_to_one`
  - `test_tide_forward_shape`
  - `test_residual_block_preserves_dim`
  - `test_tide_lookback_skip_contributes`
  - `test_init_weights_trunc_normal`
  - `test_param_mode_composite_loss_disables_cross_penalty`
- 전체 테스트 수: `41`

## 9. Dry-run 검증
- patchtst direct dry-run: `Y`
- patchtst param dry-run: `Y`
- cnn_lstm direct dry-run: `Y`
- tide direct dry-run: `Y`
- 각 모델 lower <= upper 출력 검증: `Y`

## 10. 통합 / 빌드 영향
- [`ai/train.py`](/C:/Users/user/lens/ai/train.py) 영향
  - `--band-mode` 인자 추가
  - validation/test metric 계산 전에 공통 postprocess 적용
  - dry-run에서 forward + loss smoke 수행
- [`ai/inference.py`](/C:/Users/user/lens/ai/inference.py) 영향
  - 공통 postprocess 적용
  - line head는 그대로 유지
- [`ai/loss.py`](/C:/Users/user/lens/ai/loss.py) 영향
  - `band_mode` 분기 추가
  - width loss를 `relu(upper - lower)` 기반으로 고정

## 11. CP4 준비 상태
- ticker embedding 인터페이스가 미리 준비돼 있는지: `N`
- `model_architecture.md` 결정 6건(A-3, B-2, C-2, D-1, E-1, E-2) 코드 반영: `Y`

## 12. 메모 / 생략 / 다음 단계 권고
- TiDE의 미래 covariate는 의도적으로 생략했다.
- RevIN 모듈은 normalize/denormalize를 모두 제공하지만, 현재 모델 출력은 수익률 예측이라 forward에서는 normalize 경로 중심으로 사용한다.
- CP4에서는 ticker embedding과 실제 모델별 sweep만 집중하면 된다.
