# Lens 프로젝트 상시 컨텍스트

## 역할 분담
- **나(오케스트레이터/리뷰어)**: 지시 작성 + 보고서 검증. 코드 직접 안 씀.
- **구현 에이전트**: 코드/테스트 작성. 보고서 작성.
- 보고서는 `docs/cpN_report.md` (또는 `cpN_<topic>_report.md`) 한 파일로만.
- 별도 `cpN_instruction.md` **만들지 말 것**. 지시는 에이전트 프롬프트에 직접.

## 모델 인터페이스 규약
**PatchTST / CNN-LSTM**:
```python
forward(x: Tensor[B, L, C], ticker_id: Tensor[B] | None = None) -> ForecastOutput
```
**TiDE** (future covariate 추가):
```python
forward(x: Tensor[B, L, C], ticker_id=None, future_covariate: Tensor[B, H, F] | None = None) -> ForecastOutput
```
- `C = n_features = 36` (29 base + 7 calendar). `target_channel_idx=0` (log_return) 보존, 캘린더는 끝에 붙음.
- `ForecastOutput(line, lower_band, upper_band)` — 각각 `Tensor[B, H]`.
- `num_tickers > 0`이면 `ticker_id` 필수. 누락 시 `ValueError`.
- TiDE에서 `future_cov_dim > 0`이면 `future_covariate` 필수.
- 학습/검증/추론 모든 경로에서 forward 직후 `ai.postprocess.apply_band_postprocess`로 통일 후처리.

## Calendar features (CP6.5)
`ai/preprocessing.py:build_calendar_feature_frame()`로 dataloader 단계 즉석 파생. **DB·collector·cron 무관**.
7채널: `day_of_week_sin/cos`, `month_sin/cos`, `is_month_end`, `is_quarter_end`, `is_opex_friday`.
TiDE는 미래 horizon H일치 캘린더를 `future_covariate: [B, H, 7]`로 추가 입력 받음.

## Band mode (출력 헤드)
- `band_mode="direct"`: head가 lower/upper 직접 출력. cross loss 활성.
- `band_mode="param"`: head가 (center, log_half_width) 출력 → `lower=center-exp(log_hw)`, `upper=center+exp(log_hw)`. 구조적으로 cross 불가 → cross loss 0.
- 분기 헬퍼: `ai.models.common._split_band(band_raw, band_mode)`.

## Loss (ForecastCompositeLoss)
- `line` = `AsymmetricHuberLoss(α=1, β=2)` — overprediction 페널티 큰 **보수적** 예측. α/β 변경 금지 (철학 결정).
- `band` = `PinballLoss(q=0.1)` (lower) + `PinballLoss(q=0.9)` (upper).
- `width` = `relu(upper - lower).mean()` — 음수 방지 fix 반영됨.
- `cross` = `relu(lower - upper).mean()` — `direct` 모드에서만 더함.

## RevIN (PatchTST에만)
- `revin(x, mode="norm")` → 정규화 + 통계량 캐시.
- `revin.denormalize_target(y, target_channel_idx)` → 캐시된 통계로 역정규화.
- forward 끝에서 line/lower/upper 각각 denorm 호출 필수.

## Channel Independence (PatchTST)
- `ci_aggregate ∈ {"target", "mean", "attention"}`. 기본 `"target"`.
- `target`: `target_channel_idx`(기본 0=log_return) 채널만 사용.
- `attention`: `ChannelAttentionPooling(n_features)` 학습 가중.

## Init 정책 (`ai.models.blocks.init_weights`)
- `nn.Linear`, `nn.Conv1d`: `trunc_normal_(std=0.02)`, bias=0.
- `nn.LayerNorm`: weight=1, bias=0.
- `nn.LSTM`: weight `trunc_normal_(std=0.02)`, bias=0.
- 모든 모델 `__init__` 끝에서 `self.apply(init_weights)`.

## Dropout 위치 (rate는 sweep, 위치는 고정)
- PatchTST: input dropout + encoder 내부 + output dropout.
- CNN-LSTM: conv 뒤 + LSTM 내부(n_layers>1) + attention pooling 뒤(output dropout).
- TiDE: `ResidualBlock` 내부.

## Sufficiency gate
- 1D: 보유 거래일 ≥ 450 (250 + rolling max 200).
- 1W: 보유 주봉 ≥ 78.
- 재무 피처 요구 모델: ≥ 8 분기 보유.
- 부족하면 학습/추론 모두 제외 (NaN으로 끌고 가지 말 것).

## Ticker registry
- 파일: `ai/cache/ticker_id_map_{1d,1w}.json`.
- 형식: `{"timeframe": ..., "mapping": {ticker: id, ...}, "num_tickers": N}`.
- OOV(unknown) id = `num_tickers` (즉 `Embedding(N+1, ...)` 마지막 행).
- 1D 473개 / 1W 421개 (CP3 sufficiency 통과 기준).

## 데이터 split
- 70/15/15 time-ordered (CP3 확정).
- Train → Val → Test 순서. 미래 정보 누수 방지.
- gap = h_max (=20일 기본). min_fold_samples = 50.

## 결정된 아키텍처 옵션 (model_architecture.md §결정)
- A-3 출력 정합성 (postprocess 단일 함수).
- B-2 PatchTST 논문 정확 구현 (RevIN + CI + patching + Transformer).
- C-2 CNN-LSTM (conv residual + attn pool + dilated TCN [1,2,4,8] RF=31).
- D-1 TiDE 논문 골격 (per-step head, lookback skip, future covariate 없음).
- E-1 통합 Init.
- E-2 Dropout 위치 통일.

## CP 보고서 검증 룰 (메타 D21)
모든 CP 보고서는 다음 4개 헤딩 필수:
1. 핵심 컴포넌트 존재 체크리스트 (RevIN denorm 호출, CI aggregate 종류, ticker emb 등).
2. 새 테스트 결과.
3. dry-run 결과 (lower<=upper, line_preserved).
4. 기존 회귀 통과 건수.

## 토큰 효율 룰 (메타 D24)
- 별도 instruction.md 파일 금지. 지시는 직접 프롬프트.
- 반복 컨텍스트는 본 CLAUDE.md에 박기.
- 길고 매트릭스 많은 지시는 YAML 형식 (prose 대비 40-60% 토큰).
- 보고서만 .md 파일 (검증 대상이라 파일 형태 필요).

## 한국 규제 메모 (Phase 2-3 작업 시)
- 본인 계좌 자동매매 — 합법.
- 타인 추천 알림 — 유사투자자문업 등록 필요 (자본금 1억).
- 타인 계좌 자동매매 — 투자일임업 (자본금 15억, 사실상 불가).
- 알림 표현은 사실 기술만 ("밴드 하단 돌파"), 추천 단어 ("매수/매도/지금") 금지.
