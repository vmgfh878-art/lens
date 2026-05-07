# CP4 보고서

## 1. Ticker registry
- 1D 매핑 ticker 수: 473
- 1W 매핑 ticker 수: 421
- OOV `unknown_ticker_id`: `num_tickers`
- 저장 경로:
  - `ai/cache/ticker_id_map_1d.json`
  - `ai/cache/ticker_id_map_1w.json`
- 직렬화 형식: json (`timeframe`, `mapping`, `num_tickers`)
- Roundtrip 테스트 통과: Y

## 2. F-1 RevIN denormalize 정합
- `denormalize_target(y, target_channel_idx)` 메서드 추가: Y
- PatchTST forward에서 `line/lower/upper` 각각 target channel 기준으로 역정규화: Y
- `target_channel_idx` 기본값: 0 (`log_return`)
- Roundtrip 검증: Y
- 모델 출력 raw target scale sanity:
  - PatchTST 출력 `abs(mean)`가 1.0 미만으로 유지됨
  - `NaN`, `Inf` 없음

## 3. F-2 CI 출력 정합
- `ci_aggregate` 인자 추가: Y
- `target / mean / attention` 3종 구현: Y
- `ChannelAttentionPooling` 추가: Y
- 기본값: `target`

## 4. Ticker embedding 통합
### PatchTST
- [x] `num_tickers`, `ticker_emb_dim` 인자 추가
- [x] `nn.Embedding(num_tickers + 1, ticker_emb_dim)` 추가
- [x] CI aggregate를 hidden 단계로 이동
- [x] hidden + ticker embedding concat 후 head 입력
- [x] `forward(..., ticker_id=None)` 추가
- [x] forward smoke 통과

### CNN-LSTM
- [x] `num_tickers`, `ticker_emb_dim` 인자 추가
- [x] `nn.Embedding` 추가
- [x] attention pooling 뒤에 ticker embedding concat
- [x] forward smoke 통과

### TiDE
- [x] `num_tickers`, `ticker_emb_dim` 인자 추가
- [x] `nn.Embedding` 추가
- [x] temporal decoder pool 뒤에 ticker embedding concat
- [x] lookback skip은 ticker와 무관하게 유지
- [x] forward smoke 통과

- `ticker_emb_dim`: 32
- embedding 파라미터 수:
  - 1D: 15,168
  - 1W: 13,504

## 5. F-3 Init 테스트 정제
- 측정 방식 변경(모든 큰 `nn.Linear` 집계): Y
- Tolerance 5% 적용: Y
- 모델별 std:
  - PatchTST: 0.020090
  - CNN-LSTM: 0.020474
  - TiDE: 0.019928

## 6. Dataset / Train / Inference 전파
- `ai/preprocessing.py`에서 `ticker_id` 생성 및 `SequenceDatasetBundle`에 포함: Y
- `ai/train.py`에서 `ticker_id`를 forward로 전달: Y
- `ai/inference.py`에서 `ticker_id`를 로드해 forward로 전달: Y
- CLI 인자 추가:
  - `--num-tickers`
  - `--ticker-emb-dim`
  - `--ci-aggregate`

## 7. 테스트
- 기존 AI 테스트 green: Y
- 추가 테스트:
  - ticker registry 4건
  - RevIN target denorm 3건
  - CI aggregate 3건
  - ticker embedding 5건
  - init 정밀 테스트 3건
- AI 테스트 총 46건 통과
- 백엔드 회귀 테스트 23건 통과

## 8. Dry-run 매트릭스
- `patchtst target × 1D`: Y
- `patchtst mean × 1D`: Y
- `patchtst attention × 1D`: Y
- `cnn_lstm × 1D`: Y
- `tide × 1D`: Y
- `patchtst target × 1W`: Y
- `patchtst param mode × 1D`: Y
- 모든 dry-run에서:
  - forward 통과
  - `ticker_id` batch shape 확인
  - `lower <= upper`
  - `line_preserved = true`

## 9. 통합 영향 정리
- `ai/train.py`
  - ticker embedding 관련 설정 추가
  - `ticker_id` 포함 DataLoader로 변경
  - PatchTST `ci_aggregate` 전달 추가
- `ai/inference.py`
  - checkpoint 설정에 맞는 ticker embedding 모델 로드
  - `ticker_id` 포함 추론 추가
- `ai/preprocessing.py`
  - ticker registry 저장
  - dataset에 `ticker_ids` 텐서 추가
- `ai/models/*`
  - PatchTST: target denorm, hidden aggregate, ticker embedding
  - CNN-LSTM / TiDE: concat-after-backbone ticker embedding

## 10. 메모
- 실제 학습은 아직 수행하지 않았고, dry-run과 단위 테스트만 실행했다.
- 현재 세션에는 CUDA가 보이지 않아 GPU 기준 성능 측정은 이번 CP에서 다루지 않았다.
