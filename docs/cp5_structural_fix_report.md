# TiDE per-step path
[`ai/models/tide.py`](/C:/Users/user/lens/ai/models/tide.py)에서 `decoded.mean(dim=1)`를 제거하고 `temporal_decoder -> per-step line_head/band_head` 흐름으로 바꿨다. `band_mode` 분기는 [`ai/models/common.py`](/C:/Users/user/lens/ai/models/common.py)의 `_split_band()` 헬퍼로 공통화했다.

# TiDE inherited head 처리 방식 (상속 제거 vs hidden_dim=0 등)
TiDE는 `MultiHeadForecastModel` 상속을 제거하고 `nn.Module`을 직접 상속하도록 정리했다. 그래서 상속된 `line_head/band_head` dead weight가 사라졌고, TiDE 전용 per-step head만 남는다.

# TiDE 새 테스트 결과 (2건)
`test_tide_per_horizon_path`, `test_tide_temporal_decoder_affects_output` 두 건 모두 통과했다. `last_temporal_hidden_shape`는 `(B, H, dec_dim_eff)`로 유지됐고, `temporal_decoder` 파라미터 변경 시 출력 `line`도 실제로 변했다.

# TiDE dry-run 3건 결과
`tide × 1D × direct`, `tide × 1D × param`, `tide × 1W × direct` 모두 통과했다. 세 경우 모두 `lower <= upper`, `line_preserved=true`, `ticker_id_shape=[4]`가 확인됐다.

# CNN-LSTM TCN 교체 — 채널·dilation 표
[`ai/models/cnn_lstm.py`](/C:/Users/user/lens/ai/models/cnn_lstm.py)에서 conv 블록을 `29 -> 64(d=1) -> 64(d=2) -> 64(d=4) -> 64(d=8)` 구조로 교체했다. 각 층 뒤에는 `LayerNorm + ReLU + Dropout`을 두고, 입력은 `1x1 Conv` residual로 더한다.

# CNN-LSTM 받은 receptive field (계산식 포함)
받는 receptive field는 `1 + 2*(1+2+4+8) = 31`이다. 테스트에서도 첫 시점 출력이 `t=15` 변화에는 반응하고 `t=16` 변화에는 반응하지 않아 RF=31 가정을 확인했다.

# CNN-LSTM 파라미터 수 차이 (before / after)
CNN-LSTM 파라미터 수는 `253,904 -> 278,864`로 늘었다. 증가분은 dilated conv 4층과 그에 대응하는 LayerNorm, residual 유지 비용이 반영된 결과다.

# CNN-LSTM 새 테스트 결과
`test_cnn_lstm_receptive_field`를 추가했고 통과했다. 기존 `test_cnn_lstm_forward_shape`, `test_cnn_lstm_attention_pooling_sums_to_one`도 그대로 green을 유지했다.

# CNN-LSTM dry-run 결과
`cnn_lstm × 1D × direct` dry-run이 통과했다. 출력 shape, `lower <= upper`, `line_preserved=true`, `ticker_id_shape=[4]`를 확인했다.

# 기존 AI 테스트 / backend 테스트 통과 건수
AI 테스트는 총 `49`건 통과했고, backend 회귀 테스트는 `23`건 통과했다. 이번 변경으로 PatchTST, RevIN, loss, postprocess 회귀는 새로 깨진 것이 없었다.

# 메모 / 잔여 issue
이번 CP5는 구조 fix만 반영했고 실제 학습은 아직 돌리지 않았다. dry-run 로그에서 보이는 `pandas read_sql_query` 경고는 기능 실패는 아니지만, 이후에는 SQLAlchemy 경로로 정리하는 편이 깔끔하다.
