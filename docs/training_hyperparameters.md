# Lens 학습 하이퍼파라미터 & 설계 결정 (SoT)

> **목적**: Phase 1 모델 학습에 관한 모든 파라미터·기법·설계 결정을 한 파일로 묶는다. 결정값·근거·고려한 대안·탈락 이유·민감도·Phase 2 재검토 트리거를 모두 명시. 새 결정이 생기면 **이 파일을 Edit로 업데이트** (rewrite 금지, 섹션별 append).
>
> 마지막 업데이트: 2026-04-27
> 관련 문서:
> - `.claude/plans/c-users-user-onedrive-desktop-2026-1-22-quiet-gray.md` (전체 기획안 v3)
> - [`docs/model_architecture.md`](model_architecture.md) — 모델 구조 갭 분석·개선 옵션 (CP3.5용)
> - [`docs/project_journal.md`](project_journal.md) — 진행·결정 누적 기록

---

## 목차

1. Optimizer (AdamW)
2. Learning Rate
3. Batch Size / Sequence Length
4. Gradient Clipping ★ 결정 #1
5. LR Scheduler ★ 결정 #2
6. Data Augmentation / Ensemble / Bagging ★ 결정 #3
7. Train/Val Gap ★ 결정 #4
8. Run Orchestration ★ 결정 #5
9. Validation 주기 & Primary 지표 ★ 결정 #6
10. Dropout & Weight Decay
11. Multi-seed & Early Stopping
12. Loss 설계
13. 속도 최적화 체계
14. 고정 vs Sweep 축 요약표
15. 민감도 치트시트
16. Phase 2 재검토 트리거 모음
17. CHANGELOG

---

## 1. Optimizer (AdamW)

**결정**: `torch.optim.AdamW`

**동작 개념**:
- Adam = momentum(1차 모멘트 EMA) + per-parameter adaptive LR(2차 모멘트 EMA)
- AdamW = Adam + **decoupled weight decay**. weight에 `(1 − lr·wd)` 직접 곱. L2 penalty 방식(Adam 원본)보다 adaptive LR 간섭이 없음 (Loshchilov & Hutter, 2019).

### Betas (β1, β2)

**결정**: `betas=(0.9, 0.999)` (PyTorch default, PatchTST 원논문)

| 대안 | 세팅 | 탈락 이유 |
|---|---|---|
| GPT-3 스타일 | (0.9, 0.95) | 초대형 모델·큰 배치용. Lens 규모에선 불필요 |
| momentum 강화 | (0.95, 0.999) | flat landscape엔 유리하나 금융 noise 환경엔 튐 |
| β2 낮추기 | (0.9, 0.99) | 초반 학습 빨라지지만 noise 민감 |

**민감도**: 90%의 경우 default로 충분. 튜닝 후순위.

### Weight Decay

**결정**: `weight_decay=1e-2`

| 값 | 성격 | 탈락 이유 |
|---|---|---|
| 0.0 | 정규화 없음 | overfit 방어 부족 |
| 1e-4 | 약함 (ResNet 표준) | Transformer 계열엔 부족 |
| **1e-2** | **Transformer 표준 (BERT/ViT/PatchTST)** | 채택 |
| 1e-1 | 강함 | 필요 이상 |

**민감도**: log-scale ×10 단위로 움직여야 val_loss 체감. ±10%는 의미 없음.

---

## 2. Learning Rate

**결정**: **Sweep 대상**. sweep 범위 = `[1e-4, 3e-4, 1e-3]` (log-spaced 3점)

가장 민감한 축이라 고정하지 않음. A-1 Loss sweep과 B-1 LR sweep에서 탐색.

| 값 | 체감 |
|---|---|
| 1e-5 | loss 거의 안 움직임 |
| 1e-4 | 큰 Transformer(BERT/GPT) 표준 |
| 3e-4 | Karpathy's constant, Transformer 일반 default |
| 1e-3 | Adam 전통 default, 중소 모델 |
| 1e-2 | SGD 영역. Adam엔 폭발 |

**금융 시계열 특성**: SNR 낮고 타깃 noise 크므로 비전·NLP보다 조금 낮게 시작하는 게 안전 → `1e-4`부터 넣음.

**진단**:
- 발산/NaN → LR 너무 큼
- val plateau가 초반에 깊이 → LR 너무 작음

---

## 3. Batch Size / Sequence Length

**결정**:
- `batch_size`: 128 (1D) / 64 (1W)
- `seq_len`: 252 bars (1D, 1년) / 104 bars (1W, 2년)

고정. Sweep 아님.

**근거**:
- VRAM: 5060 Ti 16GB에 PatchTST × batch=128 × seq=252 여유.
- seq_len은 기획안 §3.6 확정치. "장기 패턴 학습에 충분 + VRAM·시간 수용"의 타협점.
- batch가 너무 크면 일반화 저하("large batch 문제"), 너무 작으면 gradient noise 커서 수렴 불안정. 128은 중간점.

---

## 4. Gradient Clipping ★ 결정 #1

**결정**: `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` 고정

### 왜 필요한가

Transformer(attention) / LSTM(시간축 역전파)에서 특정 배치 gradient가 폭발적으로 튀는 현상 존재. 이걸 그대로 업데이트하면 weight가 한 step에 망가짐. Clipping = "total L2 norm이 `max_norm` 넘으면 비례 축소".

### 고려한 대안

| 방식 | 탈락 이유 |
|---|---|
| `clip_grad_value_(X)` element-wise clamp | 파라미터별 스케일 달라서 일괄 clamp가 왜곡. RL에서 드물게 사용 |
| 없음 | TiDE(MLP)는 무해하나 PatchTST·CNN-LSTM에 폭발 리스크 |

### `max_norm` 값 비교

| 값 | 성격 | 탈락 이유 |
|---|---|---|
| 0.1 | 과도한 제약, 학습 거의 정지 | 거의 안 씀 |
| 0.5 | 강함, 안정성 최우선 | 정상 gradient도 깎임 |
| **1.0** | **Transformer/LSTM 표준 (GPT, BERT, PatchTST)** | 채택 |
| 5.0 | 약한 제약, 극단 폭발만 | 폭발 방어 약함 |
| 10.0+ | 사실상 무효 | 의미 없음 |

**근거**: 정상 학습 시 total grad norm 분포는 0.5~3 범위. 1.0은 "상단만 살짝 자르는" 위치. 평소엔 발동 거의 안 함, 튀는 배치만 잡음.

**진단 방법**:
- wandb에 매 step `grad_norm_preclip` 로그
- 발동률 >20% 지속 → `max_norm` 너무 낮거나 LR 너무 큼
- 발동률 <5% → insurance policy로만 기능 (정상)

**Phase 2 재검토 트리거**: grad_norm 분포가 0.5~3 벗어나면 조정.

---

## 5. LR Scheduler ★ 결정 #2

**결정**: **Cosine annealing + linear warmup**
- `warmup_epochs = 5`
- `min_lr = max_lr × 0.01`

### 왜 scheduler가 필요한가

- 학습 초반: weight 랜덤이라 gradient noisy → 큰 LR 발산 리스크
- 중반: loss surface descent 주도, 큰 LR 유리
- 후반: 최소점 근처, 큰 LR이면 oscillation
- Constant LR은 세 구간을 하나로 때우므로 trade-off 강제됨

### 고려한 대안

| 옵션 | 탈락 이유 |
|---|---|
| (b) OneCycleLR | `max_epoch` 정확히 알아야 함, 중간 early stopping에 약함. Transformer에선 cosine+warmup 대비 이점 불명. PatchTST 원논문도 cosine |
| (c) ReduceLROnPlateau (patience=5, factor=0.5) | val_loss noisy하면 spurious trigger. Warmup 내장 없음 |
| (d) Constant | 탐색 초기 디버깅용. 최종 성능 상한 낮음. A-0 smoke test에만 한정 사용 |

### Cosine + Warmup 세부

```
epoch 0 → 5:     lr = max_lr × (epoch / 5)          선형 상승
epoch 5 → max:   cosine 곡선 max_lr → min_lr        부드러운 하강
```

**하이퍼**:
- `warmup_epochs=5`: attention head/embedding 초기 gradient 폭발 방지. Transformer 통례 (2~10 범위). max_epoch=50 기준 10% 수준.
- `min_lr = max_lr × 0.01`: 후반 미세조정 영역. 0으로 떨어뜨리면 막판 1~2 epoch 헛돌음.

**적용성**: PatchTST ✅, TiDE ✅, CNN-LSTM ✅.

**Phase 2 재검토 트리거**: max_epoch을 50에서 크게 바꾸게 되면 warmup 비율 재검토.

---

## 6. Data Augmentation / Ensemble / Bagging ★ 결정 #3

**결정**: **Phase 1에서 전부 적용 안 함**. Phase 2 대안으로만 보관.

### Data Augmentation

금융 시계열 augmentation 기법 개관:

| 기법 | 동작 | 금융 적합성 |
|---|---|---|
| Jittering | 값에 gaussian noise | 위험. 존재하지 않던 regime 생성 |
| Scaling | 전체 ×α | 수익률 의미 변질 |
| **Time warping** | 시간축 non-linear stretch | **금융 금지**. horizon 정의 깨짐 |
| Window slicing | 랜덤 부분 자르기 | sliding window와 중복 |
| Magnitude warping | 시간별 다른 스케일 | 변동성 구조 파괴 |
| Mixup | `(x1,y1),(x2,y2) → λ·x1+(1-λ)·x2` | 타임스탬프 정체성 소실 |
| **Cutout / Masking** | 일부 구간 마스킹 후 복원 self-supervised | **Phase 2 후보** |

**탈락 이유**:
1. 금융 시계열 aug 효과가 vision/NLP만큼 robust하지 않음 (Monash TSF, GluonTS 벤치에서 일관된 증거 부족).
2. 현재 병목이 "데이터 부족"인지 "capacity 과잉"인지 "SNR 한계"인지 미규명. 병목 몰라도 aug 붙이면 원인 분석 비용만 커짐.
3. 붙이면 성능 변화가 aug 때문인지 다른 변경 때문인지 분리 어려움.

### Ensemble / Bagging

- Ensemble: 여러 모델 예측 평균/stacking
- Bagging: bootstrap resampling + 모델 학습 + 평균

**탈락 이유**: 각 모델 개별 성능 평가가 먼저. Phase 1은 **multi-seed 평균** (같은 구조 × 다른 초기화 3~5개) 로 제한. 모델 간 ensemble은 Phase 2.

### Phase 2 재검토 트리거

- Phase 1 sweep에서 일관된 overfit 징후 (train↓ / val↑) → Cutout/Masking 검토
- 특정 regime sample scarcity 확인 → regime-conditional aug 검토
- 각 모델 개별 Sharpe 안정화 → stacking 도입

---

## 7. Train/Val Gap ★ 결정 #4

**결정**: `gap = h_max` (1D → 20 bars, 1W → 12 bars)

### 왜 gap이 필요한가 (누수 매커니즘)

샘플은 `(input_window, target)` 쌍. 학습 마지막 샘플 입력 끝이 `T-1`이면 타깃 범위는 `[T, T+h-1]`. 만약 validation 첫 입력이 `[T-seq_len+1, T]`에서 시작하면, 학습 타깃이 포함된 구간 `T`가 val 입력에 다시 등장 → **target-to-input leakage**. 직접 정답 노출은 아니지만 val 성능이 낙관적으로 부풀림.

### 고려한 대안

| Gap | 탈락 이유 |
|---|---|
| 0 bar | 직접 누수 |
| 1 bar (§2.4 최소치) | h=1이면 OK, h=20이면 부족 |
| **h_max (1D=20 / 1W=12)** | **모든 horizon에서 누수 차단 + 손실 <1%** (채택) |
| seq_len (252) | 파라노이드. 10% 데이터 손실 |

### 비용

- 1D 10년 ≈ 2500 거래일. gap=20 → **0.8% 손실** (무시 가능)
- 1W 10년 ≈ 520주. gap=12 → **2.3% 손실**

### 종속 변수임에 주의

Gap은 정답이 있는 결정(= h_max). 나중에 horizon 확장(예: h=60 연구)하면 gap도 자동으로 따라 올라감.

---

## 8. Run Orchestration ★ 결정 #5

**결정**: **(a) Wave=seed** 방식. 3 wave 순차 실행.

```
Wave 1: seed=42,  6 run (3 모델 × 2 TF)  → 요약 리포트 → Go/No-go 판단
Wave 2: seed=123, 6 run (동일 조합)
Wave 3: seed=777, 6 run (동일 조합)
```

### 고려한 대안

| 방식 | 탈락 이유 |
|---|---|
| (b) 모델×TF 기준 seed 3개 연속 | 한 모델 실패 시 전체 조기 중단 신호 늦음. Multi-seed 분산 집계는 빠르지만 실패 감지 느림 |
| 전체 18 run 병렬 | 5060 Ti 단일 GPU, 불가 |

### 채택 이유

- Wave 1 종료 후 6 run 결과를 보고 Wave 2/3 착수 여부 결정 → **낭비 시간 최소화**
- Run 간 의존성 없음 → W&B agent set-and-forget
- 각 wave 종료 시 CP 보고 (진행 가시성)

---

## 9. Validation 주기 & Primary 지표 ★ 결정 #6

**결정**:
- 주기: **every epoch**
- Primary 지표: **`val_pinball_loss`** (낮을수록 좋음)
- Early stopping: `patience=10`, `min_delta=1e-4`
- Best checkpoint: val_pinball_loss 최저점

### 고려한 대안

**주기 옵션**:

| 옵션 | 탈락 이유 |
|---|---|
| Every N epochs (N>1) | Early stopping 지연, ASHA rung 유연성 ↓ |
| Every N steps | W&B log 폭증, noise 많음. LLM pre-training 수준에서만 의미 |
| End only | early stopping·ASHA 둘 다 불가 |

**Primary 지표 후보**:

| 지표 | 탈락 이유 |
|---|---|
| `val_total_loss` | Loss weight sweep(A-1) 중 비교 불공정 (weight가 sweep 변수라 값 자체가 바뀜) |
| **`val_pinball_loss`** | **채택**. 밴드 본체, 기획안 §4.1 ★★★. Loss weight 무관 |
| `val_crps` | pinball과 강한 상관 → 보조 로깅 |
| `val_coverage_gap` | sharpness 무시. 보조 로깅 |

### 비용

- 1 epoch (1D, 5060 Ti, bf16+compile) ≈ 40~90초
- Val 순회(15% 크기) ≈ 6~14초 = **10~15% 오버헤드** (수용 가능)

### ASHA 결합

W&B ASHA 스케줄러 rung = [5, 10, 20, 40] epoch. every-epoch val이면 모든 rung에서 자연스레 유효. every-N-epoch은 N이 rung 배수여야 함 → 유연성 떨어짐.

### Phase 2 재검토 트리거

1 epoch이 5분 이상 걸리는 환경(대형 유니버스/긴 seq_len) → `every 2 epochs`로 축소 검토.

---

## 10. Dropout & Weight Decay

**결정** (모델별 고정):
- PatchTST: `dropout=0.2`
- CNN-LSTM: `dropout=0.3`
- TiDE: `dropout=0.1`

각 원논문 범위. Sweep 대상 아님 (기획안 §3.5 확정).

Weight decay는 §1 참고 (`1e-2` 전 모델 공통).

---

## 11. Multi-seed & Early Stopping

**Multi-seed**: 3 seed (42, 123, 777) 기본. 필요 시 5개까지 확장.

**근거**: 딥러닝 결과의 re-run 분산은 무시 못 함. 3 seed 평균으로 "모델 A가 B보다 낫다"를 주장할 최소 근거 확보. 5 seed까지 가면 신뢰도↑이지만 run 비용 ×1.67 → Phase 1은 3.

**Early stopping**: `patience=10`, `min_delta=1e-4`, monitor = `val_pinball_loss`.

- patience=10: val이 10 epoch 동안 개선 없으면 종료. Transformer 통례 10~20 중 짧은 쪽 (빠른 반복).
- min_delta=1e-4: 너무 민감하면 noise로 계속 재시작, 너무 둔감하면 과학습. 1e-4가 pinball loss 스케일 기준 합리적.

---

## 12. Loss 설계

**구성** (`ai/loss.py::ForecastCompositeLoss`):

| 컴포넌트 | 구현 | 역할 |
|---|---|---|
| Line head | `AsymmetricHuberLoss(α=1, β=2, δ=1.0)` | 점 예측선. β>α로 과대예측 페널티 |
| Band head | `PinballLoss((q_low, q_high))` 두 개 합 | q10/q90 밴드 |
| Width penalty | `WidthPenaltyLoss` | 밴드 폭 과도 확장 억제 |
| Cross penalty | `BandCrossPenaltyLoss` | 하단 > 상단 위반 방지 |

**합성**:
```
total = λ_line · line + λ_band · (q_low + q_high) + λ_width · width + λ_cross · cross
```

**기본값**: `λ_line=1.0, λ_band=1.0, λ_width=0.1, λ_cross=1.0`

**Sweep 대상**: Loss weight 4축 → A-1 sweep에서 coverage·sharpness trade-off 탐색.

### 기획안 §3.4.2 와의 차이 (기록)

기획안은 line head에 `AsymmetricBCELoss`를 명시했으나, BCE는 [0,1] 확률 대상이라 **수익률 타깃에 부적합**. 실제 구현은 `AsymmetricHuberLoss` 사용. α=1, β=2 비대칭은 유지 (과대예측 페널티 철학 일치).

### Quantile crossing 처리

**결정 (잠정)**: `torch.sort`로 quantile 후처리 강제 정렬 (`ai/inference.py::sort_prediction_triplet`).

대안:
- Loss penalty (cross_loss): 이미 composite에 포함
- 후처리 정렬: 추론 시점 추가 보장

두 개 병용. 기획안 §11.2 보류 항목 A는 3주차 학습 진입 시 재판단 예정.

---

## 13. 속도 최적화 체계

### 티어 1 — 반드시 켠다 (CP1 반영)

| 기법 | 동작 | 이득 |
|---|---|---|
| **bf16 autocast** | forward·backward만 16-bit (brain-float). weight는 fp32 유지. fp16과 달리 지수부 8-bit로 NaN 안전 → GradScaler 불필요 | 2~2.5× (Blackwell) |
| **torch.compile(reduce-overhead)** | 연산 그래프 trace → Triton 커널 fuse | 1.3~1.8× (Transformer) |
| **TF32 matmul** (`set_float32_matmul_precision("high")`) | fp32 대신 19-bit TF32 | matmul 1.5~2× |
| **DataLoader** num_workers=6 + pin_memory + persistent_workers | CPU 병렬 + GPU 전송 최적화 | CPU 병목 제거 |
| **Flash Attention (SDPA)** | PyTorch 2.x `F.scaled_dot_product_attention` | Attention 2~4×, 메모리↓ |
| **Feature tensor 캐시** | `ai/cache/features_{tf}_{hash}.pt` | preprocessing 재계산 제거 |

### 티어 2 — 조건부, **Phase 1 적용 안 함**

| 기법 | 언제 쓰나 | Lens 판단 |
|---|---|---|
| Gradient Accumulation | VRAM 부족 | 5060 Ti 16GB로 충분, 불필요 |
| fp16 + GradScaler | bf16 미지원 구형 GPU | Blackwell은 bf16 우선, 불필요 |
| Channels-last | CNN 많을 때 5~15% | CNN-LSTM만 해당, 복잡도 대비 이득 소소 |
| CUDA graphs | 극단적 launch overhead | `torch.compile`이 내부에서 일부 활용, 별도 불필요 |
| DeepSpeed / FSDP | 모델이 GPU에 안 들어감 | 무관 |
| Dataset RAM caching | 작은 데이터셋 | 이미 `.pt` 캐시로 해결 |

### 속도 체감 추정

CP1 CPU baseline (AAPL 1D PatchTST batch=128 1epoch) = 2.074s
GPU(5060 Ti) bf16+compile 추정: CPU 대비 6~10× 기본 + 1.5~2× 추가 = **0.1~0.2s/epoch** 상대

풀 유니버스 기준:
- PatchTST 1D 1 epoch ≈ 40~90초
- 50 epoch × 6 run (1 wave) ≈ 3.5~8시간
- 3 wave ≈ **10~24시간** (W&B set-and-forget)

실측은 CP7에서 확정.

---

## 14. 고정 vs Sweep 축 요약

| 파라미터 | 값 | 성격 |
|---|---|---|
| Optimizer | AdamW | 고정 |
| betas | (0.9, 0.999) | 고정 |
| weight_decay | 1e-2 | 고정 |
| **learning_rate (max)** | **sweep**: [1e-4, 3e-4, 1e-3] | A-1/B-1 sweep |
| batch_size | 128 (1D) / 64 (1W) | 고정 |
| seq_len | 252 (1D) / 104 (1W) | 고정 |
| gradient clipping max_norm | 1.0 | 고정 (#1) |
| scheduler | Cosine + warmup 5 | 고정 (#2) |
| min_lr ratio | 0.01 | 고정 |
| augmentation / ensemble | 없음 | 고정 — Phase 2 이관 (#3) |
| train/val gap | h_max | 고정 (#4) |
| run orchestration | Wave=seed ×3 | 고정 (#5) |
| val 주기 | every epoch | 고정 (#6) |
| val primary metric | val_pinball_loss | 고정 (#6) |
| early stopping patience | 10 | 고정 |
| min_delta | 1e-4 | 고정 |
| max_epochs | 50 | 고정 (early stopping이 실제 종료 결정) |
| dropout | 모델별 고정 | 고정 |
| seeds | {42, 123, 777} | 고정 (3개) |
| **loss weights λ** | **sweep** | A-1 sweep |

---

## 15. 민감도 치트시트

"얼마나 움직이면 어떻게 변하나" 직감표. 처음 튜닝할 때 참고.

| 파라미터 | 작게 | 크게 | 체감 scale |
|---|---|---|---|
| learning_rate | plateau 유지 | NaN/발산 | log ×10 |
| weight_decay | overfit | underfit | log ×10 |
| dropout | overfit | underfit | ±0.1 |
| batch_size | noisy gradient, 일반화↑ | 일반화↓, GPU 효율↑ | ×2 |
| seq_len | 장기 패턴 놓침 | 학습 느려짐, 메모리↑ | ×2 |
| grad clip max_norm | 학습 정체 | explosion 못 잡음 | ×2 |
| warmup_epochs | 초기 발산 | 수렴 지연 | ±2 |
| patience | 과조기 종료 | 과학습 | ±5 |

---

## 16. Phase 2 재검토 트리거 모음

| 현상 | 재검토 항목 |
|---|---|
| grad_norm 분포가 0.5~3 벗어남 | max_norm 조정 |
| train↓ val↑ 일관된 overfit | Cutout/Masking augmentation 검토 |
| 각 모델 Sharpe 안정 | Model stacking ensemble 검토 |
| 1 epoch > 5분 | val 주기 축소 (every 2 epochs) |
| Horizon 확장 (h=60+) | train/val gap 자동 확장 |
| 대형 유니버스 전환 | batch_size·VRAM 재계산 |
| TiDE만 성능 열위 지속 | 아키텍처 제외 검토 |

---

## 17. CHANGELOG

- **2026-04-24**: 최초 작성. 결정 #1~#6 모두 확정 반영.
  - #1 gradient clipping max_norm=1.0
  - #2 Cosine + warmup 5
  - #3 augmentation/ensemble/bagging 전부 Phase 2 이관
  - #4 train/val gap = h_max
  - #5 Wave=seed × 3
  - #6 every epoch val + val_pinball_loss primary

---

## 18. 2026-04-27 CP10~CP12 기준 최신 학습 계약

이 섹션은 CP10 이후의 최신 운영 기준이다. 위 §9와 §12는 당시 설계 기록으로 유지하되, 다음 CP 발주와 실험 해석은 이 섹션을 우선한다.

### 18.1 공통 평가판

CP10 이후 모델 평가는 다음 지표 묶음을 공통으로 기록한다.

| 묶음 | 지표 | 의미 |
|---|---|---|
| 방향 | `direction_accuracy` | target 유형에 따라 threshold 분리. `direction_label`은 0.5, raw return 계열은 0.0 |
| 랭킹 | `spearman_ic`, `top_k_long_spread`, `top_k_short_spread`, `long_short_spread` | 같은 날짜 단면에서 종목 선별력이 있는지 확인 |
| 밴드 | `coverage`, `avg_band_width`, `band_loss` | AI band의 calibration과 sharpness 확인 |
| 오차 | `mae`, `smape` | MAPE 대체. 0 근처 수익률에서 MAPE 왜곡 회피 |
| 비용 반영 | `fee_adjusted_return`, `fee_adjusted_sharpe`, `fee_adjusted_turnover` | 수수료 이후에도 남는 신호인지 확인 |

투자 지표는 학습 target과 무관하게 항상 `raw_future_returns` 기준으로 계산한다. 이 원칙이 깨지면 volatility-normalized target이나 direction_label 실험에서 투자 성과가 잘못 해석된다.

### 18.2 target plumbing

현재 지원 상태는 다음과 같다.

| target | 상태 | 비고 |
|---|---|---|
| `raw_future_return` | 구현 | 가격 decode, 저장, 시그널 생성 가능 |
| `volatility_normalized_return` | 구현 | 평가는 raw realized return 기준. inference는 score_only |
| `direction_label` | 구현 | direction_accuracy는 0.5 기준. 저장/시그널 생성 차단 |
| `market_excess_return` | 자리만 있음 | 다음 target CP에서 SPY/시장 수익률 조인 필요 |
| `rank_target` | 자리만 있음 | 다음 target CP에서 날짜 단면 랭킹 라벨 필요 |

비 raw target 체크포인트는 price decode를 허용하지 않는다. `--save`와 BUY/SELL/HOLD 시그널 생성도 차단한다.

### 18.3 최신 loss 계약

기본 forecast 손실은 line과 band를 유지한다.

```text
forecast_loss = lambda_line * line_loss
              + lambda_band * band_loss
              + lambda_cross * cross_loss
```

CP11 direction head를 켠 경우에만 보조 손실을 더한다.

```text
total_loss = forecast_loss + lambda_direction * direction_loss
```

현재 기본값과 의미는 다음과 같다.

| 파라미터 | 기본 | 의미 |
|---|---|---|
| `lambda_line` | 1.0 | 하방 보수적 line 유지 |
| `lambda_band` | 1.0 | quantile band 본체 유지 |
| `lambda_cross` | 1.0 | lower/upper crossing 방지 |
| `lambda_direction` | 0.1 | 방향 보조 head만 약하게 반영 |
| `lambda_width` | 레거시 | CLI 호환용. 현재 손실 계산에는 사용하지 않음 |

direction target은 데이터셋 target을 바꾸지 않고 `raw_future_returns > 0`에서 내부 생성한다. 하락을 상승으로 잘못 예측하는 쪽이 더 위험하므로 target=0 가중치 2.0, target=1 가중치 1.0을 사용한다.

### 18.4 모델 채택 guardrail

direction head, rank target, market excess target 실험은 다음 guardrail을 통과해야 한다.

| 영역 | 통과 기준 |
|---|---|
| 정체성 | line은 하방 보수적이어야 하고, band 품질이 본체로 유지돼야 함 |
| 밴드 | coverage가 목표 구간에서 무너지지 않고, avg_band_width가 과도하게 넓어지지 않아야 함 |
| 보수성 | overprediction_rate와 mean_overprediction이 악화되면 채택 보류 |
| 투자성 | Spearman IC, top-k spread, fee-adjusted return을 함께 확인 |
| 방향 | direction_accuracy 단독 개선만으로 채택 금지 |

CP11 결과 기준으로 CNN-LSTM direction head는 5티커 smoke에서는 좋아 보였지만 50티커에서는 확실한 이득이 아니었다. 따라서 채택 보류가 맞다.

### 18.5 precision과 finite gate

CP12 이후 precision 옵션은 다음과 같다.

| 옵션 | 의미 |
|---|---|
| `--amp-dtype bf16` | 기본값. 기존 CUDA mixed precision 경로 유지 |
| `--amp-dtype fp16` | 필요 시 비교용 |
| `--amp-dtype off` | 모든 phase에서 autocast 미사용 |
| `--detect-anomaly` | backward NaN 원인 추적용. 느리므로 기본 off |

finite gate는 train, validation, test, checkpoint 저장 전 단계에 적용된다. NaN/Inf가 나오면 정상 결과로 저장하지 않는다.

| 결과 | 처리 |
|---|---|
| 정상 종료 | `model_runs.status="completed"` |
| NaN/Inf 실패 | `model_runs.status="failed_nan"` 메타만 저장 |
| failed_nan run | predictions, prediction_evaluations, backtest_results, checkpoint 저장 차단 |

NaN run을 완전히 숨기지 않는 이유는 디버그 재현성 때문이다. 언제, 어떤 config, 어떤 phase/metric/batch에서 깨졌는지 남겨야 다음 CP에서 원인을 줄일 수 있다.

### 18.6 CP12 매트릭스 이후 분기

CP12 보고서 8번 섹션의 6개 명령을 실행한 뒤 다음처럼 판단한다.

| 결과 | 다음 조치 |
|---|---|
| CUDA amp off PASS, bf16 NaN | 부분 fp32 강제 옵션 검토 |
| 50티커 bf16 baseline PASS | CP13 direction sweep과 PatchTST/TiDE 미러 가능 |
| direction head만 NaN | direction_logit 또는 BCEWithLogits 경로 분리 |
| baseline도 NaN | 모델 성능 실험 중지. precision/runtime 안정성 우선 |

### 18.7 CHANGELOG 추가

- **2026-04-27**: CP10~CP12 기준 최신 학습 계약 추가. 공통 평가판, raw realized return 기준, 비 raw target score_only, direction 보조 loss, finite gate, failed_nan 저장 정책, amp dtype 분기를 SoT로 명시.

### 18.8 CP12 매트릭스 결과와 CP12.5 학습 정책

CP12 매트릭스 결과, CUDA에서 autocast를 끄면 CNN-LSTM baseline과 direction head가 통과했고, bf16 autocast를 켜면 baseline과 direction head 모두 val `avg_band_width`에서 NaN이 발생했다.

| 비교 | 결과 | 해석 |
|---|---|---|
| CPU fp32 baseline | PASS | 데이터와 기본 모델 경로는 정상 |
| CUDA amp off baseline | PASS | CUDA 자체는 가능 |
| CUDA bf16 baseline | NAN | bf16 autocast 경로 문제 유력 |
| CUDA amp off direction_head | PASS | direction head 자체가 NaN 원인은 아님 |
| CUDA bf16 direction_head | NAN | baseline과 같은 bf16 문제 |
| 50티커 CUDA bf16 baseline | NAN | 소규모 우연이 아니라 실제 sweep 조건에서 재현 |

CP12.5 전까지의 운영 정책:

- CNN-LSTM CUDA bf16 결과는 성능 판단에 쓰지 않는다.
- direction head, rank target, market excess target 확대는 중단한다.
- `--amp-dtype off` 결과만 임시로 신뢰할 수 있지만, `[2]`, `[4]` 종료코드 1이 있어 정상 종료 경로까지 확인해야 한다.
- CP12.5에서 `--fp32-modules` 옵션을 도입해 `lstm`, `heads`, `conv`, `postprocess/eval` 중 어느 구간이 bf16 NaN을 만드는지 분리한다.

CP12.5 성공 조건:

| 조건 | 기준 |
|---|---|
| 안정성 | 5티커 CUDA bf16 baseline PASS |
| 규모 재현성 | 50티커 CUDA bf16 baseline PASS |
| direction 경로 | 5티커 CUDA bf16 direction_head PASS |
| 진단성 | 실패 시 first NaN tensor/metric/module이 보고됨 |
| 프로세스 상태 | PASS 케이스 종료코드 0 |

### 18.9 CP12.5 결과와 임시 학습 정책

CP12.5 결과로 `--fp32-modules lstm,heads`가 CNN-LSTM CUDA bf16 NaN을 지표상 해결하는 후보로 확인됐다.

| 옵션 | 결과 | 정책 |
|---|---|---|
| `none` | NAN | 사용 금지 |
| `heads` | NAN | 사용 금지. head가 1차 원인이 아님 |
| `lstm` | PASS | 최소 안정화 후보 |
| `lstm,heads` | PASS | 다음 검증 기본 후보 |

단, 모든 CUDA PASS 케이스가 프로세스 종료 시 `-1073740791`로 끝났다. 따라서 현재 정책은 다음과 같다.

- 성능 실험은 계속 보류한다.
- CNN-LSTM CUDA bf16 실험은 `--fp32-modules lstm,heads`를 붙였을 때만 지표를 볼 수 있다.
- 그래도 종료코드가 0이 아니므로 sweep/Optuna/W&B 정식 trial로 인정하지 않는다.
- CP12.6에서 성공 run 종료코드 0을 확보해야 CP13으로 넘어갈 수 있다.

CP12.6에서 우선 분리할 후보:

| 후보 | 확인 이유 |
|---|---|
| CUDA context teardown | JSON 출력 후 죽으므로 종료 시점 cleanup 가능성 |
| DataLoader / pin_memory / worker cleanup | Windows CUDA 종료에서 자주 섞이는 비정상 종료 후보 |
| W&B / stdout flush / logging finalization | `--no-wandb`에서도 재현되는지 재확인 필요 |
| torch compile / dynamo | 현재 `--no-compile` 조건에서도 재현됐으므로 1차 원인은 아니지만 env flag 재확인 |
| OpenMP / MKL 충돌 | 기존 `KMP_DUPLICATE_LIB_OK` 계열 이슈와 연관 가능성 |

CP12.6 성공 기준:

- 5티커 CUDA amp off baseline 종료코드 0.
- 5티커 CUDA bf16 `--fp32-modules lstm,heads` baseline 종료코드 0.
- 50티커 CUDA bf16 `--fp32-modules lstm,heads` baseline 종료코드 0.
- 실패 시 Python traceback인지 native crash인지 구분된 보고.

### 18.10 CP12.6 closure와 PatchTST 단일 학습 정책

CP12.6에서 CNN-LSTM CUDA 종료코드 문제는 cuDNN LSTM 경로로 분리되어 해결됐다. 따라서 런타임 신뢰성 CP는 닫는다. 그러나 Phase 1 학습 정책은 멀티모델 병렬 개선이 아니라 PatchTST 단일 주력으로 전환한다.

Phase 1 학습 기본값:

| 항목 | 값 |
|---|---|
| 주력 모델 | `patchtst` |
| 비교 모델 | Phase 1에서는 freeze. TiDE/CNN-LSTM은 Phase 1.5 |
| target 우선순위 | `raw_future_return` baseline → `volatility_normalized_return` → `market_excess_return` |
| primary 판단 | band guardrail + IC/top-k/fee-adjusted 지표 |
| 금지 | direction_accuracy 단독 최적화, 세 모델 동시 미러링, 긴 sweep 선행 |

PatchTST sweep 우선순위:

| 순서 | 축 | 이유 |
|---|---|---|
| 1 | target type | 투자 목적과 학습 목적 일치가 먼저 |
| 2 | `patch_len`, `stride` | PatchTST 핵심 설계축 |
| 3 | `seq_len` | 긴 lookback 활용 여부 확인 |
| 4 | `ci_aggregate` | target/mean/attention 중 Lens 단일 target에 맞는 결합 방식 확인 |
| 5 | `d_model`, `n_layers`, `dropout` | 구조 용량은 마지막에 조정 |

임시 보류:

- PatchTST direction head.
- PatchTST rank head.
- self-supervised pretraining.
- TiDE/CNN-LSTM 동시 적용.

이 보류는 기능 폐기가 아니라 순서 조정이다. 제품 루프와 PatchTST 기준선이 안정화되면 Phase 1.5 또는 후속 research CP에서 다시 연다.

### 18.11 CP13 closure와 CP14 학습 진입 기준

CP13에서 PatchTST Solo Track 실험 축을 확정했다. CP14는 다음 기준을 따른다.

실행 가능한 target baseline:

| target | 실행 여부 | 이유 |
|---|---|---|
| `raw_future_return` | 실행 | 기준선 |
| `volatility_normalized_return` | 실행 | 구현 완료. 투자 지표는 raw realized return 기준으로 계산됨 |
| `market_excess_return` | 보류 | 시장 벤치마크 조인 경로 미구현 |
| `rank_target` | 보류 | 단면 랭킹 생성 경로 미구현 |
| `direction_label` | 보류 | 이번 단계는 direction 분류 실험이 아님 |

CP14 선행 결정:

- `patch_len`, `stride`, `d_model`, `n_heads`, `n_layers`를 `ai.train` CLI에 직접 노출할지, PatchTST 전용 sweep entry로만 열지 결정한다.
- Stage A target baseline은 현재 CLI만으로 바로 실행한다.
- Stage B patch geometry와 Stage E capacity 조정은 인자가 열린 뒤에만 실행한다.

CP14 baseline은 `--save-run` 없이 먼저 실행해 지표만 비교한다. checkpoint는 생성되지만 DB run 저장은 하지 않는다. baseline 후보가 정해진 뒤에만 `--save-run`으로 대표 run을 남긴다.

### 18.12 Research CP와 Product CP 병렬 운영

PatchTST 학습은 Research CP에서만 변경한다. Product CP는 저장된 run과 평가/백테스트 결과를 읽어서 보여주는 역할만 맡는다.

운영 원칙:

- Product CP는 target/loss/model architecture를 변경하지 않는다.
- Research CP는 프론트 레이아웃과 API 응답 형태를 변경하지 않는다.
- Product 화면에서 보여줄 지표는 Research CP의 guardrail과 동일하게 둔다.
- 모델 파라미터 조정 UI는 Phase 1에서는 read-only config panel로만 구현한다.
- 화면에서 학습 실행 버튼을 만들지 않는다.

Product 화면에 노출할 학습 파라미터:

| 묶음 | 항목 |
|---|---|
| target | `line_target_type`, `band_target_type` |
| PatchTST geometry | `seq_len`, `patch_len`, `stride` |
| PatchTST capacity | `d_model`, `n_heads`, `n_layers`, `dropout` |
| optimization | `lr`, `weight_decay`, `batch_size`, `epochs`, `seed` |
| status | `run_id`, `status`, `best_epoch`, `best_val_total`, `checkpoint_path` |

target type 노출 원칙:

- 주식 보기 첫 화면에는 target type을 전면 노출하지 않는다.
- target type은 모델 학습 화면과 run 상세에서 보여준다.
- 일반 사용자 언어는 `AI 밴드`, `보수적 예측선`, `가격만 제공`을 사용한다.
- `volatility_normalized_return`은 점수형 target이므로 가격 decode/시그널 생성 가능 모델처럼 보이지 않게 주의한다.

### 18.13 학습 시간 ETA와 GPU 사용량 운영

Research CP에서 모델 학습 명령을 발주할 때는 실행 시간 예상을 같이 제공한다.

기본 기록 항목:

| 항목 | 기록 방식 |
|---|---|
| batch size | 명령 인자와 실제 적용값 |
| VRAM | 실행 중 최대 사용량 |
| GPU util | 가능하면 평균 또는 체감 범위 |
| epoch time | 첫 epoch 실제 시간 |
| 총 ETA | `첫 epoch 시간 * 남은 epoch` 기준으로 재계산 |
| 중단 기준 | NaN/OOM/지표 악화/시간 초과 기준 |

PatchTST batch size 운영:

| 단계 | 목적 |
|---|---|
| `batch_size=64` | 안전 기준선 |
| `batch_size=128` | 16GB VRAM 활용 1차 상향 |
| `batch_size=256` | 안정적일 때만 처리량 상향 후보 |

VRAM을 가득 채우는 것이 목적은 아니다. 목표는 안정성을 유지하면서 epoch 시간을 줄이고, 성능 지표가 흔들리지 않는 batch size를 찾는 것이다. 16GB 환경에서는 OS와 CUDA 여유를 남기기 위해 대략 8~12GB 사용을 1차 목표로 둔다.

### 18.14 제품 화면과 학습 지표 연결 원칙

CP16-P 이후 제품 화면은 최신 완료 PatchTST run의 prediction과 evaluation을 읽어 `AI 밴드`, `보수적 예측선`, `커버리지`, `평균 밴드 폭`을 보여준다.

운영 원칙:

- 제품 화면은 학습 파라미터를 바꾸지 않는다.
- 주식 보기 화면은 `completed` run만 사용한다.
- `failed_nan` run은 모델 학습 화면의 상태 추적에만 남긴다.
- 주식 보기 화면에는 target type을 노출하지 않는다.
- 모델 학습 화면에서만 `line_target_type`, `band_target_type`, `seq_len`, `batch_size`, `lr` 같은 실험 설정을 보여준다.
- 제품 화면의 지표는 Research CP의 guardrail과 같은 의미를 유지한다.

### 18.15 CP17-P 이후 지표 표시 계약

CP17-P 이후 백테스트 화면과 모델 학습 화면은 latest completed PatchTST run의 `val_metrics`/`test_metrics`를 함께 보여준다.

우선 표시 지표:

| 묶음 | 지표 |
|---|---|
| 밴드 | `coverage`, `avg_band_width` |
| 오차 | `mae`, `smape` |
| 랭킹/투자 | `spearman_ic`, `top_k_long_spread`, `long_short_spread` |
| 비용 반영 | `fee_adjusted_return`, `fee_adjusted_sharpe`, `fee_adjusted_turnover` |
| 보조 | `direction_accuracy` |

표시 원칙:

- `coverage`, `avg_band_width`를 위쪽에 배치한다.
- `direction_accuracy`는 보조 위치에 둔다.
- 백테스트 화면에서는 수수료 반영 지표와 수수료 전 gross 지표를 분리한다.
- 주식 보기 화면에는 여전히 target type을 노출하지 않는다.

### 18.16 데모 산출물 readiness 원칙

제품 데모에서 AI overlay와 백테스트 지표를 보여주려면 저장 산출물이 필요하다.

필수 산출물:

| 산출물 | 필요 이유 |
|---|---|
| completed `model_runs` | 제품 화면이 기준 run을 선택하기 위해 필요 |
| `predictions` row | 주식 보기 AI 밴드/보수적 예측선 overlay 표시 |
| `prediction_evaluations` row | ticker/asof 평가 요약 표시 |
| `backtest_results` row | 백테스트 화면 표시 |

원칙:

- fake data를 넣지 않는다.
- prediction/backtest/evaluation이 없으면 empty state로 보여준다.
- 발표 전에는 실제 inference/backtest 절차로 데모용 completed run 산출물을 확보한다.
- readiness check는 `AAPL 1D`만 고집하지 말고, 실제 prediction이 저장된 ticker/run을 기준으로 선택할 수 있어야 한다.

### 18.17 PatchTST 전체 학습 시간 기준과 smoke-first 원칙

PatchTST 입력 feature finite contract가 복구된 뒤, `raw_future_return` 1D 전체 학습은 NaN 없이 완료됐다. 하지만 `473 ticker × seq_len 252 × batch 64 × 5 epoch` 조건은 epoch당 약 800~930초, 총 약 80분이 걸렸다.

따라서 이후 Research CP는 다음 원칙을 따른다.

| 단계 | 기본값 | 목적 |
|---|---|---|
| finite smoke | `--limit-tickers 50 --epochs 1 --batch-size 64` | 새 target/구조가 NaN 없이 도는지 확인 |
| throughput check | `batch_size 64/128/256` | 16GB VRAM 활용과 epoch time 확인 |
| short compare | `--limit-tickers 50 --epochs 3~5` | 지표 방향성 확인 |
| full confirm | 전체 ticker, 5 epoch 이상 | 후보가 좁혀진 뒤 최종 확인 |

금지:

- 새 target이나 구조를 전체 473 ticker 5 epoch로 바로 실행하지 않는다.
- W&B 없이 긴 학습을 하염없이 기다리지 않는다.
- 첫 epoch 실제 시간이 나오기 전 총 소요 시간을 단정하지 않는다.

ETA 기준:

- 전체 473 ticker, batch 64, seq_len 252는 1 epoch 약 13~16분으로 본다.
- 5 epoch는 약 70~90분으로 본다.
- 50 ticker smoke는 이보다 훨씬 짧아야 하며, 이 값으로 batch ladder와 full run ETA를 다시 계산한다.

### 18.18 CP19 데모 산출물 기준

현재 데모용 실제 산출물 기준은 다음이다.

| 항목 | 값 |
|---|---|
| demo run | `patchtst-1D-fc096a026a1e` |
| demo ticker | `AAPL` |
| timeframe | `1D` |
| horizon | `5` |
| forecast series length | `5` |

이 run은 최신 completed run은 아니지만, 실제 prediction/evaluation/backtest row가 존재한다. 제품 화면은 최신 completed run부터 확인하되, prediction row가 없는 경우 최근 completed run 중 사용 가능한 prediction run을 선택한다.

주의:

- demo run의 지표는 발표용 작동 확인용이지 최종 성능 주장 근거가 아니다.
- 성능 주장은 이후 smoke-first 실험과 full confirm run을 통해 다시 정리한다.
- fake data는 사용하지 않는다.
### CP21-R PatchTST geometry smoke 기준

PatchTST 실험은 full run 전에 50티커 smoke로 먼저 확인한다.

공통 smoke 조건:

| 항목 | 값 |
|---|---|
| model | `patchtst` |
| timeframe | `1D` |
| seq_len | `252` |
| epochs | `3` |
| batch_size | `256` |
| target | `raw_future_return` |
| ci_aggregate | `target` |
| compile | off |
| W&B | off |
| limit_tickers | `50` |

geometry 비교 결과:

| 조건 | patch_len | stride | n_patches | 평균 epoch_seconds | VRAM peak |
|---|---:|---:|---:|---:|---:|
| baseline | 16 | 8 | 30 | 75.9685 | 7781 MB |
| short | 8 | 4 | 62 | 136.2211 | 13464 MB |
| dense | 16 | 4 | 60 | 132.1239 | 13039 MB |
| long | 32 | 16 | 14 | 45.6483 | 4976 MB |
| overlap | 32 | 8 | 28 | 71.5938 | 7472 MB |

현재 기본 후보:

- 주력: `patch_len=16`, `stride=8`
- 보조 calibration 후보: `patch_len=32`, `stride=16`

주의:

- coverage가 0.98 이상이면 경고 구간으로 본다.
- coverage 1.0은 좋은 결과로 단정하지 않는다.
- geometry 변경만으로 밴드 폭과 투자 지표를 동시에 개선하지 못했으므로, 다음 단계는 band calibration을 우선한다.
### CP22-R band calibration smoke 기준

PatchTST geometry는 `patch_len=16`, `stride=8`로 유지한다. CP22-R에서는 q 범위, `lambda_band`, `band_mode`만 비교했다.

현재 calibration 기준점:

| 항목 | 값 |
|---|---|
| patch_len | 16 |
| patch_stride | 8 |
| band_mode | `direct` |
| q_low | 0.20 |
| q_high | 0.80 |
| lambda_band | 2.0 |

이 설정은 50티커 3epoch에서 `coverage=0.972607`, `avg_band_width=0.303344`, `spearman_ic=0.069890`, `long_short_spread=0.005254`, `fee_adjusted_return=6.956047`을 기록했다.

주의:

- coverage가 아직 0.98에 가깝기 때문에 최종 후보는 아니다.
- 다음 smoke에서는 `q_low=0.25~0.30`, `q_high=0.70~0.75` 범위를 우선 본다.
- `band_mode=param`은 이번 smoke에서 투자 지표가 크게 약해졌으므로 우선순위를 낮춘다.
### CP23-R narrow band 후보

CP23-R 결과 기준으로 PatchTST band calibration 후보를 갱신한다.

1차 후보:

| 항목 | 값 |
|---|---|
| q_low | 0.30 |
| q_high | 0.70 |
| lambda_band | 2.0 |
| band_mode | `direct` |
| patch_len | 16 |
| patch_stride | 8 |

50티커 3epoch 결과:

| metric | 값 |
|---|---:|
| coverage | 0.890662 |
| lower_breach_rate | 0.030712 |
| upper_breach_rate | 0.078626 |
| avg_band_width | 0.214104 |
| band_loss | 0.067403 |
| spearman_ic | 0.069014 |
| long_short_spread | 0.004678 |
| fee_adjusted_return | 4.653222 |

보수 후보:

| 항목 | 값 |
|---|---|
| q_low | 0.25 |
| q_high | 0.75 |
| lambda_band | 2.0 |

보수 후보는 coverage가 0.936846으로 높지만 투자 지표 보존이 가장 좋다. CP24-R의 Bollinger 비교 전까지는 `q30-b2`와 `q25-b2`를 100티커 또는 1W smoke 후보로 두었다.

### CP24-R Bollinger 비교 후 calibration 후보

CP24-R에서 Bollinger 기준선을 같은 `raw_future_return` validation/test split에 맞춰 계산했다. Bollinger는 가격 밴드가 아니라 horizon별 완료 수익률의 rolling 평균과 표준편차를 사용한다.

현재 PatchTST 후보:

| 역할 | q_low | q_high | lambda_band | band_mode | 판단 |
|---|---:|---:|---:|---|---|
| 보수형 | 0.25 | 0.75 | 2.0 | `direct` | coverage 0.936846. BB60-2.0s와 비슷하지만 폭은 넓다. |
| 기본형 | 0.30 | 0.70 | 2.0 | `direct` | coverage 0.890662. BB20-2.0s와 BB20-1.5s 사이에 있어 1차 후보로 둔다. |
| 공격형 | 0.35 | 0.65 | 2.0 | `direct` | upper breach 0.112994. BB20-1.5s와 비슷하므로 과도하게 좁다고 단정하지 않는다. |

다음 100티커 안정성 확인 후보는 `q30-b2`와 `q35-b2`다. `q25-b2`는 conservative reserve로 남긴다.

### CP25-R 100티커 안정성 결과

100티커 확장 smoke에서 `q30-b2`, `q35-b2`는 모두 validation band guardrail을 통과하지 못했다.

| 후보 | q_low | q_high | lambda_band | coverage | upper_breach_rate | avg_band_width | 판단 |
|---|---:|---:|---:|---:|---:|---:|---|
| q30-b2 | 0.30 | 0.70 | 2.0 | 0.463702 | 0.283601 | 0.069402 | 기본 후보 탈락 |
| q35-b2 | 0.35 | 0.65 | 2.0 | 0.384560 | 0.323423 | 0.056351 | 공격 후보 탈락 |

현재 full run 후보는 없다.

다음 검증 순서:

1. `q25-b2`: `q_low=0.25`, `q_high=0.75`, `lambda_band=2.0`
2. q25도 실패하면 `q20-b2`: `q_low=0.20`, `q_high=0.80`, `lambda_band=2.0`
3. 이후에도 validation/test 격차가 크면 walk-forward 또는 ticker group split을 먼저 적용한다.

### CP26-R 100티커 recalibration 결과

100티커 기준에서 q25/q20/q15 계열을 다시 확인했다.

최종 checkpoint 기준:

| 후보 | q_low | q_high | lambda_band | coverage | upper_breach_rate | avg_band_width | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| q25-b2 | 0.25 | 0.75 | 2.0 | 0.544565 | 0.242948 | 0.083824 | 탈락 |
| q20-b2 | 0.20 | 0.80 | 2.0 | 0.650025 | 0.188960 | 0.104741 | 탈락 |
| q15-b2 | 0.15 | 0.85 | 2.0 | 0.749335 | 0.137470 | 0.128752 | 공격 후보 |
| q20-b1 | 0.20 | 0.80 | 1.0 | 0.652703 | 0.186760 | 0.105156 | 탈락 |

현재 상태:

- 기본형 preset은 아직 없다.
- 공격형 preset은 `q15-b2`만 남긴다.
- full run은 금지한다.

중요한 학습 정책 이슈:

- 현재 checkpoint selection은 validation total loss 기준이다.
- 이번 CP에서는 total loss가 내려갈수록 밴드가 좁아져 coverage가 나빠졌다.
- 다음에는 coverage-aware checkpoint selection 또는 `q15-b2` 2epoch 재현 smoke를 먼저 수행한다.

### CP28-R 200티커 coverage-gate 결과

CP27-R에서 추가한 `--checkpoint-selection coverage_gate`는 유지한다. 다만 200티커 확장 결과 q20/q25/q15 후보 모두 eligible checkpoint가 없어 `coverage_gate_failed_fallback_val_total`로 떨어졌다.

| preset | q_low | q_high | lambda_band | selected_epoch | coverage | upper_breach_rate | avg_band_width | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| q20-b2 | 0.20 | 0.80 | 2.0 | 3 | 0.598411 | 0.224333 | 0.093772 | 탈락 |
| q25-b2 | 0.25 | 0.75 | 2.0 | 3 | 0.499277 | 0.276971 | 0.074906 | 탈락 |
| q15-b2 | 0.15 | 0.85 | 2.0 | 3 | 0.702103 | 0.166290 | 0.116981 | 탈락 |

현재 학습 정책:

- `checkpoint_selection=coverage_gate`는 유지한다.
- 위 세 preset은 200티커 기준 full run 후보로 올리지 않는다.
- full 473티커는 금지한다.
- 다음 실험은 PatchTST preset 미세조정보다 DLinear/NLinear baseline 또는 calibration 재설계가 우선이다.
### CP29-D 이후 학습 데이터 계약

CP29-D부터 학습 입력 feature contract version은 `v3_adjusted_ohlc`다. 이 버전은 가격 파생 피처를 adjusted OHLC 기준으로 다시 계산한다.

고정된 계약:

- `open_ratio/high_ratio/low_ratio`는 raw OHLC가 아니라 adjusted OHLC 기준이다.
- DB `indicators`에 오래된 가격 ratio가 남아 있어도 `fetch_training_frames`에서 `price_data`를 사용해 가격 파생 피처를 덮어쓴다.
- `vol_change`의 `Inf`는 제거하고 finite contract를 통과한 샘플만 학습에 사용한다.
- 50/100/200 v3 feature cache 모두 feature contract 이후 non-finite count 0을 확인했다.

CP29-D에서 허용한 학습은 50티커 1epoch smoke뿐이다. `--save-run`은 사용하지 않았고 full 473티커 학습은 실행하지 않았다. smoke 명령은 `patchtst`, `1D`, `seq_len=252`, `horizon=5`, `epochs=1`, `batch_size=256`, `limit_tickers=50`, `patch_len=16`, `patch_stride=8`, `q_low=0.20`, `q_high=0.80`, `lambda_band=2.0`, `band_mode=direct`, `checkpoint_selection=coverage_gate` 조합이다.

### CP30-G 이후 저장/재현 게이트

CP30-G에서는 학습 성능 비교를 재개하지 않았다. full 473티커, W&B sweep, 대형 비교 실험은 계속 금지한다.

학습 저장 계약:

| 항목 | 값 |
|---|---|
| feature version | `v3_adjusted_ohlc` |
| band mode 저장 | `model_runs.band_mode` 정식 컬럼 |
| coverage gate 통과 | `status=completed` |
| coverage gate 실패 fallback | `status=failed_quality_gate` |
| NaN/Inf 실패 | `status=failed_nan` |

`failed_quality_gate`는 학습 프로세스가 끝났고 checkpoint와 metric은 남길 수 있지만, 제품 inference/backtest 대상은 아니다. 따라서 completed run 목록에는 섞지 않는다.

prediction 저장 계약:

- `predictions` upsert key는 `run_id,ticker,model_name,timeframe,horizon,asof_date`다.
- 같은 ticker/date라도 run이 다르면 서로 덮어쓰지 않는다.
- `prediction_evaluations`는 `lower_breach_rate`, `upper_breach_rate`를 정식 컬럼으로 저장한다.

inference 계약:

- ticker embedding checkpoint는 checkpoint의 `ticker_registry_path`를 사용한다.
- registry mismatch는 실패 처리한다.
- subset inference가 ticker id를 다시 매기지 않는다.
- 학습 registry는 ticker 집합 fingerprint가 포함된 파일로 저장한다.

backtest 계약:

- anchor는 `adjusted_close` 기준이다.
- position과 turnover는 날짜별 포트폴리오 단위다.
- active `BUY/SELL` 포지션은 절대 노출 합 1로 정규화한다.

RevIN은 이번 CP에서 변경하지 않는다. 다음 ablation은 같은 split, 같은 seed, 같은 CP29/CP30 데이터/저장 계약에서 `use_revin=True`와 `use_revin=False`만 비교한다.

### CP31-S 저장 run 스모크 결과

저장 계약 검증용 학습은 성능 실험이 아니라 다음 최소 조건으로 실행했다.

```powershell
python -m ai.train --model patchtst --timeframe 1D --seq-len 252 --epochs 1 --batch-size 64 --device cuda --no-wandb --no-compile --ci-aggregate target --line-target-type raw_future_return --band-target-type raw_future_return --limit-tickers 5 --patch-len 16 --patch-stride 8 --band-mode direct --save-run
```

저장 run 정책 확인:

- `model_runs.feature_version`은 `v3_adjusted_ohlc`다.
- `model_runs.band_mode`는 `direct`로 저장된다.
- `model_runs.status`는 completed run이면 `completed`, coverage gate fallback이면 `failed_quality_gate`다.
- inference는 `model_runs.checkpoint_path`와 checkpoint 내부 `ticker_registry_path`를 사용한다.
- `predictions`와 `prediction_evaluations`는 run_id를 unique key에 포함해 run별 이력을 보존한다.
- `backtest_results`는 `run_id,strategy_name,timeframe` 단위로 저장한다.

CP31-S 실제 run에서는 `patchtst-1D-01ab0fc075f4`가 completed로 저장됐고, inference/backtest 저장까지 통과했다. 같은 ticker/asof_date에 두 번째 run `patchtst-1D-239b58ab90f0`을 저장해도 기존 prediction row가 덮이지 않는 것을 확인했다.

### CP32-M clean feature 재검증 결과

CP32-M은 기존 PatchTST preset을 `feature_version=v3_adjusted_ohlc` 기준으로 다시 검증했다. 조건은 `seq_len=252`, `patch_len=16`, `patch_stride=8`, `ci_aggregate=target`, `raw_future_return`, `checkpoint_selection=coverage_gate`, `batch_size=256`, `--no-compile`, `--no-wandb`로 고정했다.

50티커 결과:

| preset | q_low | q_high | lambda_band | selected_reason | coverage | upper_breach_rate | test fee | 판정 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| baseline | 0.10 | 0.90 | 1.0 | coverage_gate_failed_fallback_val_total | 0.999977 | 0.000012 | -0.333283 | 탈락 |
| q20-b2 | 0.20 | 0.80 | 2.0 | coverage_gate_failed_fallback_val_total | 0.984291 | 0.011219 | -0.266205 | 탈락 |
| q25-b2 | 0.25 | 0.75 | 2.0 | coverage_gate_eligible | 0.949298 | 0.035584 | -0.103182 | 탈락 |
| q15-b2 | 0.15 | 0.85 | 2.0 | coverage_gate_failed_fallback_val_total | 0.995788 | 0.002738 | -0.222129 | 탈락 |

현재 정책:

- 위 preset들은 100티커 이상으로 올리지 않는다.
- full 473티커는 금지한다.
- `coverage_gate`는 유지하지만, clean feature 기준에서는 기존 q preset만으로 유효 후보가 없다.

### CP33-M RevIN ablation 결과

PatchTST RevIN 실험을 위해 `--use-revin / --no-use-revin`을 학습 CLI에 노출했다. 기본값은 기존 동작 보존을 위해 `--use-revin`이다.

50티커 3epoch 결과:

| preset | use_revin | q_low | q_high | lambda_band | selected_reason | coverage | upper_breach_rate | test fee | 판정 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| q25-b2 | true | 0.25 | 0.75 | 2.0 | coverage_gate_eligible | 0.949298 | 0.035584 | -0.103182 | 탈락 |
| q25-b2 | false | 0.25 | 0.75 | 2.0 | coverage_gate_failed_fallback_val_total | 0.992308 | 0.004687 | -0.484688 | 탈락 |
| q30-b2 | false | 0.30 | 0.70 | 2.0 | coverage_gate_failed_fallback_val_total | 0.980636 | 0.011045 | -0.376120 | 탈락 |
| q35-b2 | false | 0.35 | 0.65 | 2.0 | coverage_gate_failed_fallback_val_total | 0.962583 | 0.021186 | -0.454023 | 탈락 |

현재 정책:

- PatchTST 기존 preset 실험은 중단한다.
- full 473티커는 계속 금지한다.
- 다음 CP는 같은 clean feature와 저장 계약 위에서 DLinear/NLinear baseline 준비로 전환한다.

### CP34-M 예측선/밴드 평가판 분리와 q25-b2 smoke

CP34-M에서는 성능 비교판을 예측선, 밴드, 시그널/백테스트로 분리했다. 시그널/백테스트 지표는 아직 규칙 기반이 단순하므로 모델 탈락 기준으로 과하게 쓰지 않고 보조 지표로만 기록한다.

공통 학습 조건:

| 항목 | 값 |
|---|---|
| feature_version | `v3_adjusted_ohlc` |
| timeframe | `1D` |
| seq_len / horizon | `252 / 5` |
| line_target_type / band_target_type | `raw_future_return / raw_future_return` |
| checkpoint_selection | `coverage_gate` |
| batch_size | `256` |
| compile / wandb / save_run | off / off / off |
| ticker limit / epochs | 50 / 3 |
| q preset | q25-b2, `q_low=0.25`, `q_high=0.75`, `lambda_band=2.0` |

결과:

| 모델 | selected_epoch | selected_reason | coverage | upper_breach_rate | avg_band_width | spearman_ic | long_short_spread | 판정 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| TiDE | 3 | coverage_gate_failed_fallback_val_total | 0.426697 | 0.235190 | 0.040327 | -0.019393 | 0.000855 | 탈락 |
| CNN-LSTM | 3 | coverage_gate_failed_fallback_val_total | 0.631953 | 0.231825 | 0.053698 | -0.065226 | -0.003284 | 탈락 |

현재 정책:

- q25-b2 기준 TiDE와 CNN-LSTM은 밴드 후보로 채택하지 않는다.
- TiDE/CNN-LSTM 자체를 최종 폐기한 것은 아니며, 필요하면 q20-b2 같은 더 넓은 preset만 최소 smoke로 재확인한다.
- PatchTST는 line 후보로 보류하되 direct band 후보로는 현재 탈락/보류한다.
- full 473티커 학습은 계속 금지한다.

### CP35-M TiDE/CNN-LSTM rescue smoke 설정

CP35-M에서는 TiDE/CNN-LSTM을 바로 폐기하지 않고, 최소 rescue smoke만 수행했다. 공통 조건은 `feature_version=v3_adjusted_ohlc`, `1D`, `horizon=5`, `raw_future_return`, `coverage_gate`, `batch_size=256`, `epochs=3`, `limit_tickers=50`, `--no-compile`, `--no-wandb`, `--save-run` 미사용이다.

| 실험 | 모델 | seq_len | q_low | q_high | lambda_band | band_mode | selected_reason | coverage | upper_breach_rate | avg_band_width | 판정 |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| A | TiDE | 252 | 0.15 | 0.85 | 2.0 | direct | coverage_gate_failed_fallback_val_total | 0.618807 | 0.152117 | 0.065644 | 탈락 |
| B | TiDE | 252 | 0.10 | 0.90 | 2.0 | direct | coverage_gate_failed_fallback_val_total | 0.726848 | 0.102320 | 0.087572 | 근접 탈락 |
| C | CNN-LSTM | 120 | 0.20 | 0.80 | 2.0 | direct | coverage_gate_failed_fallback_val_total | 0.720917 | 0.163246 | 0.064094 | 근접 탈락 |
| D | CNN-LSTM | 60 | 0.20 | 0.80 | 2.0 | direct | coverage_gate_failed_fallback_val_total | 0.727251 | 0.169876 | 0.062560 | 근접 탈락 |
| E | TiDE | 252 | 0.10 | 0.90 | 2.0 | param | coverage_gate_failed_fallback_val_total | 0.823158 | 0.062084 | 0.107054 | 밴드 약한 보류 |

`positive_width`는 현재 CLI에 없으므로 E는 `band_mode=param`으로 실행했다. `param`은 `center ± exp(log_half_width)` 방식이라 밴드 폭을 양수로 강제한다.

현재 정책:

- TiDE q10-b2 param은 band 전용 관점에서만 약한 보류 후보다.
- line 기준은 A~E 모두 실패했다.
- 현재 selector가 line 지표까지 포함하므로, band 후보를 계속 보려면 `coverage_gate`를 band 전용 selector와 line-aware selector로 분리해야 한다.
- CP35 결과만으로 full 473티커 학습에 들어가지 않는다.

### CP36-M checkpoint selection 모드

CP36-M부터 `--checkpoint-selection`은 다음 값을 받는다.

| 값 | 사용 목적 | 상태 |
|---|---|---|
| `val_total` | legacy loss 기준 | 유지 |
| `line_gate` | 예측선 실험 | 신규 |
| `band_gate` | 밴드 실험 | 신규 |
| `combined_gate` | line + band 동시 실험 | 신규 |
| `coverage_gate` | 기존 명령 호환 | deprecated alias |

`line_gate`는 `spearman_ic > 0`, `long_short_spread > 0`, `mae finite`, `smape finite`를 요구한다. 정렬은 `spearman_ic desc`, `long_short_spread desc`, `mae asc`, `forecast_loss asc` 순서다.

`band_gate`는 `0.75 <= coverage <= 0.95`, `upper_breach_rate <= 0.15`, `lower_breach_rate <= 0.20`, `avg_band_width > 0`, `band_loss finite`를 요구한다. 정렬은 목표 coverage 0.85와의 거리, upper breach, lower breach, band width, band loss 순서다.

`combined_gate`는 두 gate를 모두 통과해야 한다. 기존 `coverage_gate`는 같은 정책의 alias로 남긴다.

CP36 smoke:

| 실험 | selector | role | gate_failed | line_gate_pass | band_gate_pass | combined_gate_pass |
|---|---|---|---:|---:|---:|---:|
| TiDE q10-b2 param | band_gate | band_model | false | true | true | true |
| PatchTST q25-b2 | line_gate | line_model | false | true | false | false |

현재 정책상 `band_gate` 통과 run은 line 실패와 무관하게 band 모델로 저장 가능하다. 반대로 `line_gate` 통과 run은 coverage 실패와 무관하게 line 모델로 저장 가능하다.

### CP37-M 역할별 재평가 결과

CP37-M은 50티커 3epoch, `v3_adjusted_ohlc`, `1D`, `horizon=5`, `batch_size=256`, `--no-compile`, `--no-wandb`, `--save-run` 미사용 조건으로 실행했다.

| 실험 | 모델 | selector | seq_len | q_low | q_high | band_mode | selected_epoch | gate_pass | 판정 |
|---|---|---|---:|---:|---:|---|---:|---|---|
| A | PatchTST | line_gate | 252 | 0.25 | 0.75 | direct | 2 | line true | line 생존 |
| B | TiDE | band_gate | 252 | 0.10 | 0.90 | param | 3 | band true | band 보류 |
| C | TiDE | band_gate | 252 | 0.10 | 0.90 | direct | 1 | band true | band 보류 |
| D | CNN-LSTM | band_gate | 120 | 0.20 | 0.80 | direct | 3 | band false | 탈락 |
| E | CNN-LSTM | band_gate | 60 | 0.20 | 0.80 | direct | 2 | band true | band 보류 |

주요 지표:

| 실험 | coverage | upper_breach_rate | spearman_ic | long_short_spread | test coverage | test upper_breach_rate |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.981634 | 0.013528 | 0.073986 | 0.005177 | 0.980075 | 0.015325 |
| B | 0.823158 | 0.062084 | -0.027770 | -0.001251 | 0.589787 | 0.135763 |
| C | 0.776830 | 0.070205 | -0.038396 | -0.001571 | 0.529882 | 0.157517 |
| D | 0.720917 | 0.163246 | -0.039287 | -0.000196 | 0.662821 | 0.191043 |
| E | 0.755517 | 0.142050 | -0.043676 | -0.000364 | 0.698842 | 0.157765 |

현재 정책:

- PatchTST는 line model 후보로 유지한다.
- TiDE param/direct와 CNN-LSTM seq60은 band model 보류 후보로 둔다.
- band 후보는 line IC/spread 음수만으로 탈락시키지 않는다.
- test coverage가 낮으므로 full 473티커는 여전히 금지한다.

### CP38-M band calibration smoke

CP38-M에서는 CP37 band 후보 중 TiDE param과 CNN-LSTM seq60을 대상으로 후처리 calibration을 적용했다. 추가 학습은 하지 않고 checkpoint 예측을 재사용했다.

공통 calibration 설정:

| 항목 | 값 |
|---|---|
| target_coverage | 0.85 |
| 대상 split | validation으로 calibration fit, test로 평가 |
| 학습 loss 변경 | 없음 |
| save-run | 없음 |

결과:

| 후보 | 방식 | test coverage | test upper_breach_rate | test lower_breach_rate | test avg_band_width | 판정 |
|---|---|---:|---:|---:|---:|---|
| TiDE param | 원본 | 0.589787 | 0.135763 | 0.274450 | 0.105727 | 탈락 |
| TiDE param | scalar width | 0.648969 | 0.140620 | 0.210411 | 0.116588 | 탈락 |
| TiDE param | conformal residual | 0.598300 | 0.130814 | 0.270886 | 0.126687 | 탈락 |
| CNN-LSTM seq60 | 원본 | 0.699801 | 0.157572 | 0.142628 | 0.082385 | 탈락 |
| CNN-LSTM seq60 | scalar width | 0.829036 | 0.078392 | 0.092571 | 0.128096 | 생존 |
| CNN-LSTM seq60 | conformal residual | 0.708474 | 0.152734 | 0.138792 | 0.111810 | 탈락 |

채택 후보:

- `CNN-LSTM seq60 q20-b2 direct + scalar width calibration`
- lower_scale: 1.095193
- upper_scale: 1.420661

현재 정책:

- band 후보 1순위는 CNN-LSTM seq60 + scalar width calibration이다.
- TiDE param은 validation/test 분포 이동이 커서 단순 calibration으로는 보류 해제하지 않는다.
- 다음은 full run이 아니라 100티커 제한 검증이다.

### CP39-M 100티커 역할 후보 검증

CP39-M은 100티커 3epoch, `v3_adjusted_ohlc`, `1D`, `horizon=5`, `batch_size=256`, `--no-compile`, `--no-wandb`, `--save-run` 미사용 조건으로 실행했다.

CNN-LSTM band 설정:

| 항목 | 값 |
|---|---|
| model | `cnn_lstm` |
| seq_len | 60 |
| q_low / q_high | 0.20 / 0.80 |
| lambda_band | 2.0 |
| band_mode | `direct` |
| checkpoint_selection | `band_gate` |
| fp32_modules | `lstm,heads` |

CNN-LSTM 결과:

| 상태 | coverage | lower_breach_rate | upper_breach_rate | avg_band_width | 판정 |
|---|---:|---:|---:|---:|---|
| raw validation | 0.672702 | 0.175678 | 0.151621 | 0.048305 | 실패 |
| raw test | 0.573707 | 0.264400 | 0.161893 | 0.053179 | 실패 |
| scalar validation | 0.850022 | 0.074968 | 0.075010 | 0.087036 | 통과 |
| scalar test | 0.796509 | 0.112616 | 0.090876 | 0.098413 | 통과 |

scalar width calibration 계수:

- lower scale: 1.908173
- upper scale: 1.378499

PatchTST line 설정:

| 항목 | 값 |
|---|---|
| model | `patchtst` |
| seq_len | 252 |
| patch_len / patch_stride | 16 / 8 |
| q_low / q_high | 0.25 / 0.75 |
| lambda_band | 2.0 |
| checkpoint_selection | `line_gate` |

PatchTST 결과:

| split | spearman_ic | long_short_spread | mae | smape | line_gate |
|---|---:|---:|---:|---:|---|
| validation | 0.018231 | 0.002256 | 0.049054 | 1.525516 | 통과 |
| test | 0.045832 | 0.006690 | 0.055652 | 1.508459 | 통과 |

현재 정책:

- line 후보는 PatchTST q25-b2 line_gate다.
- band 후보는 CNN-LSTM seq60 q20-b2 direct + scalar width calibration이다.
- full 473티커는 아직 금지한다.
- 다음은 200티커 제한 검증 또는 조합 저장 계약 스모크다.

## CP40-M 조합 inference smoke 설정

CP40-M에서는 학습을 새로 돌리지 않고 CP39 checkpoint를 사용해 조합 prediction 계약만 검증했다.

| 항목 | 값 |
|---|---|
| line checkpoint | `patchtst_1D_patchtst-1D-19103d294e6b.pt` |
| band checkpoint | `cnn_lstm_1D_cnn_lstm-1D-b882658d1561.pt` |
| split | `test` |
| tickers | `A`, `AAPL`, `ABBV`, `ABNB`, `ABT` |
| horizon | 5 |
| device | `cpu` |
| amp_dtype | `off` |
| batch_size | 256 |
| composition_version | `line_band_v1` |

Band calibration 계수는 CP39 100티커 결과를 그대로 사용했다.

| 항목 | 값 |
|---|---:|
| lower_scale | 1.908173 |
| upper_scale | 1.378499 |
| target_coverage | 0.85 |

Smoke 검증 결과:

| 항목 | 값 |
|---|---|
| row_count | 5 |
| lower <= upper | PASS |
| series length = 5 | PASS |
| 필수 meta 포함 | PASS |
| coverage | 0.920000 |
| avg_band_width | 0.290036 |

이번 CP의 수치는 성능 판단이 아니라 저장/추론 계약 검증용이다.

## CP41-S 실제 저장 run 조합 smoke

CP41-S에서는 `--save-run`으로 실제 `model_runs.run_id`를 만든 뒤 composite 저장을 실행했다.

지시된 5티커/1epoch strict 조건은 gate 통과가 되지 않아 저장 계약 입력으로 사용할 수 없었다. 최종 composite에는 `completed` 상태 run만 사용했다.

| 역할 | 조건 | run_id | status |
|---|---|---|---|
| line_model | PatchTST q25-b2, 50티커, 1epoch, `line_gate` | `patchtst-1D-41d584bcb3cb` | completed |
| band_model | CNN-LSTM seq60 q20-b2, 50티커, 3epoch, `band_gate`, `fp32_modules=lstm,heads` | `cnn_lstm-1D-76f363b84218` | completed |
| composite | A/AAPL/ABBV/ABNB/ABT 5 row 저장 smoke | `composite-1D-a0786769a07a` | completed |

Composite 실행 설정:

| 항목 | 값 |
|---|---|
| split | `test` |
| device | `cuda` |
| amp_dtype | `off` |
| batch_size | 256 |
| lower_scale | 1.908173 |
| upper_scale | 1.378499 |
| save | true |

저장 결과:

| 항목 | 값 |
|---|---:|
| predictions rows | 5 |
| prediction_evaluations rows | 5 |
| backtest_results rows | 1 |
| coverage | 0.800000 |
| avg_band_width | 0.354500 |
| upper_breach_rate | 0.160000 |

Backtest는 composite signal이 `HOLD`라 `num_trades=0`, `return_pct=0`이다. 이는 성능 결과가 아니라 저장 계약 확인값이다.

## CP42-M 200티커 안정성 검증 설정

PatchTST line 200티커 설정:

| 항목 | 값 |
|---|---|
| model | `patchtst` |
| checkpoint_selection | `line_gate` |
| seq_len | 252 |
| patch_len / patch_stride | 16 / 8 |
| q_low / q_high | 0.25 / 0.75 |
| lambda_band | 2.0 |
| batch_size | 256 |
| epochs | 3 |
| limit_tickers | 200 |
| save_run | false |

PatchTST 결과:

| split | spearman_ic | long_short_spread | line_gate |
|---|---:|---:|---|
| validation | 0.069564 | 0.012140 | PASS |
| test | 0.028802 | 0.001868 | 참고 |

CNN-LSTM band 200티커 설정:

| 항목 | 값 |
|---|---|
| model | `cnn_lstm` |
| checkpoint_selection | `band_gate` |
| seq_len | 60 |
| q_low / q_high | 0.20 / 0.80 |
| lambda_band | 2.0 |
| band_mode | `direct` |
| fp32_modules | `lstm,heads` |
| batch_size | 256 |
| epochs | 3 |
| limit_tickers | 200 |
| save_run | false |

CNN-LSTM scalar calibration:

| 항목 | 값 |
|---|---:|
| lower_scale | 2.731744 |
| upper_scale | 1.767985 |
| validation coverage | 0.849999 |
| test coverage | 0.830066 |
| test upper_breach_rate | 0.085802 |
| test lower_breach_rate | 0.084131 |
| test avg_band_width | 0.119297 |

속도 참고:

| 모델 | epoch seconds | VRAM peak |
|---|---|---:|
| PatchTST | 310.44 / 309.09 / 301.35 | 약 5153 MB |
| CNN-LSTM | 103.05 / 96.18 / 98.82 | 약 324 MB |

현재 full run 전제 후보는 `PatchTST line + CNN-LSTM calibrated band`다. 다만 full 473 진입 전 composite policy를 먼저 결정해야 한다.

### CP43-M composite 후처리 정책

CP43-M은 학습 하이퍼파라미터 변경이 아니라 composite inference 후처리 정책 비교다. 재학습 없이 CP42의 200티커 test split 예측을 사용했다.

고정값:

| 항목 | 값 |
|---|---|
| line model | PatchTST q25-b2 |
| band model | CNN-LSTM seq60 q20-b2 direct |
| band calibration | scalar width |
| lower_scale | 2.731744 |
| upper_scale | 1.767985 |
| split | test |
| row_count | 188 |

채택 후보:

| 정책 | coverage | upper breach | lower breach | avg width | line_inside |
|---|---:|---:|---:|---:|---:|
| risk_first_lower_preserve | 0.870213 | 0.122340 | 0.007447 | 0.294814 | 1.000000 |

다음 저장 smoke에서는 `composition_policy=risk_first_lower_preserve`를 meta에 남겨야 한다. full 473티커 실행은 아직 금지다.

### CP44-D ATR ratio 백필과 학습 설정 영향

CP44-D에서는 학습을 실행하지 않았다. `atr_ratio`는 indicator-only 컬럼으로 백필했으며, 학습 feature 설정은 그대로 유지한다.

| 항목 | 값 |
|---|---|
| `MODEL_N_FEATURES` | 36 |
| `atr_ratio in FEATURE_COLUMNS` | false |
| `atr_ratio in MODEL_FEATURE_COLUMNS` | false |
| feature contract version | 변경 없음 |
| cache 무효화 | 없음 |
| 모델 학습 | 실행 안 함 |

백필 후 indicator 분포는 1D p99 0.073267, 1W p99 0.163629, 1M p99 0.367066이다. max는 1D SMCI 0.240202, 1W CVNA 1.096561, 1M CVNA 8.468068로 확인됐다. 따라서 나중에 `atr_ratio`를 학습 피처로 추가할 경우 1W/1M 극단값에 대한 clipping 또는 winsorization을 먼저 정해야 한다.

이번 CP의 테스트는 feature/output contract와 collector timeframe 제한만 확인했다. full 473티커 모델 run, W&B sweep, 대형 성능 비교는 실행하지 않았다.

### CP44-M composite 저장 기본값

Composite 저장 경로 기본 정책은 `risk_first_lower_preserve`로 둔다.

| 항목 | 값 |
|---|---|
| CLI | `--composition-policy risk_first_lower_preserve` |
| 기본값 | `risk_first_lower_preserve` |
| lower 정책 | `min(calibrated_lower, line)` |
| upper 정책 | `max(calibrated_upper, line)` |
| 목적 | 하방 리스크를 축소하지 않고 line을 밴드 안에 포함 |

CP44-M save-run smoke에서는 `composition_run_id=composite-1D-3a44b5e51ed2`, coverage 0.920000, upper breach 0.080000, line_inside_band_ratio 1.000000을 기록했다. full 473티커 실행은 아직 금지이며, 다음 단계는 CNN-LSTM band sweep이다.

### CP46-M composite upper calibration

CP46-M에서는 학습 하이퍼파라미터를 바꾸지 않고 composite 후처리 상단 보정만 비교했다.

고정값:

| 항목 | 값 |
|---|---|
| band checkpoint | `cnn_lstm_1D_cnn_lstm-1D-01f09b20945a.pt` |
| lower_scale | 1.869896 |
| upper_scale | 1.513046 |
| base policy | `risk_first_lower_preserve` |

채택 후보:

| 항목 | 값 |
|---|---|
| composition policy | `risk_first_upper_buffer_1.10` |
| upper buffer scale | 1.10 |
| test coverage | 0.856383 |
| test upper breach | 0.143617 |
| test lower breach | 0.000000 |
| width increase ratio | 1.087218 |

`risk_first_upper_buffer_1.25`는 coverage 0.926596으로 과보수라 기본값으로 두지 않는다. `asymmetric_quantile_expand`는 validation에서는 통과했지만 test에서 upper breach 0.257447로 실패했다.

### CP48-M h20 smoke hyperparameters

CP48-M에서는 h20 feasibility만 확인했으며 full run이나 save-run은 금지했다.

| 항목 | PatchTST line h20 | CNN-LSTM band h20 |
|---|---|---|
| timeframe | 1D | 1D |
| horizon | 20 | 20 |
| seq_len | 252 | 252 |
| limit_tickers | 50 | 50 |
| epochs | 1 | 1 |
| batch_size | 256 | 256 |
| target | raw_future_return | raw_future_return |
| checkpoint_selection | line_gate | band_gate |
| q_low/q_high | 0.25/0.75 | 0.15/0.85 |
| lambda_band | 2.0 | 2.0 |
| band_mode | direct | direct |
| amp_dtype | bf16 | bf16 |
| compile | off | off |
| W&B | off | off |
| save-run | off | off |

결과상 PatchTST h20은 `line_gate_failed_fallback_val_total`, CNN-LSTM h20은 raw 기준 `band_gate_failed_fallback_val_total`이었다. CNN-LSTM h20 band는 scalar width calibration으로 test coverage 0.801636까지 회복됐지만, composite에서는 upper breach 0.268750으로 불안정했다.

h20을 다시 실험할 경우 기본값을 본류에 섞지 말고 별도 branch에서 `epochs >= 3`, W&B on, h20 전용 report 이름을 사용한다.

### CP49-M PatchTST horizon rescue hyperparameters

CP49-M 실행판의 공통 조건은 다음과 같다. 실제 W&B matrix 실행은 사용자가 VSCode 로컬에서 수행한다.

| 항목 | 값 |
|---|---|
| model | patchtst |
| timeframe | 1D |
| target | raw_future_return |
| checkpoint_selection | line_gate |
| limit_tickers | 50 |
| epochs | 3 |
| batch_size | 256 |
| compile | off |
| save-run | off |
| W&B | matrix 실행 시 on |

실험 후보는 h5/h10/h20 각각 patch_len/stride 16/8, 32/16, 16/4이며, h20에는 seq_len 504 baseline 후보를 추가한다. severe downside 기준은 h1~h5 5%, h6~h10 8%, h11~h20 12%로 둔다.

CP49-M matrix 결과에서 다음 설정을 우선 후보로 둔다.

| 목적 | horizon | seq_len | patch_len | stride | 상태 |
|---|---:|---:|---:|---:|---|
| 기본 line | 5 | 252 | 32 | 16 | 생존 |
| 기존 기준선 | 5 | 252 | 16 | 8 | 보류 |
| risk-only line | 5 | 252 | 16 | 4 | 보류 |
| h20 branch | 20 | 252 | 32 | 16 | Phase 1.5 보류 |
| h20 seq504 | 20 | 504 | 16 | 8 | 탈락 |

다음 100티커 재확인은 h5 longer_context 32/16부터 실행한다. h20은 false_safe_rate를 낮추는 selector 또는 보수 line 전용 selector가 생기기 전까지 본류로 올리지 않는다.

### CP51-M 평가 지표 하이퍼파라미터

CP51-M은 학습 하이퍼파라미터를 바꾸지 않았다. 평가 지표 계산에만 다음 기준을 추가했다.

| 항목 | 기본값 | 설명 |
|---|---:|---|
| `nominal_coverage` | `q_high - q_low` | 후보별 quantile 폭에서 계산 |
| `interval_lower_penalty_weight` | 2.0 | 하방 breach를 상방보다 더 크게 벌점 |
| `interval_upper_penalty_weight` | 1.0 | 상방 breach 벌점 |
| `interval_score` | width + penalty | 밴드 폭과 breach를 같이 평가 |
| `squeeze_breakout_rate` | width 하위 20% 기준 | 좁은 밴드 구간에서 realized move가 큰 비율 |

W&B key는 기존과 같이 `train/<metric>`, `val/<metric>`, `test/<metric>` 형태를 유지한다. 새 지표는 모두 scalar로 기록되며, JSON schema는 `docs/cp51_metric_schema.json`을 기준으로 한다.

### CP52-M 메트릭 기준선 계약

CP52-M은 학습 인자를 바꾸지 않았고, 평가 기준만 고정했다.

| 항목 | 값 |
|---|---|
| severe downside threshold | train split raw future return q10 |
| squeeze breakout threshold | train split abs(raw future return) q80 |
| interval lower penalty weight | 2.0 |
| interval upper penalty weight | 1.0 |
| coverage 판단 | `coverage` 단독이 아니라 `coverage_abs_error` 기준 |
| quantile calibration | lower/upper만 지원. q50은 별도 head 전까지 미지원 |

v1 guideline:

| 역할 | 생존 기준 | 후보 기준 |
|---|---|---|
| line IC mean | > 0 | >= 0.02 |
| line IC IR | > 0.25 | >= 0.5 |
| long_short_spread net | > 0 | >= 0.003 |
| fee_adjusted_sharpe | > 0 | >= 0.3 |
| false_safe_tail_rate | < 0.25 | < 0.15 |
| severe_downside_recall | >= 0.65 | >= 0.80 |
| downside_capture_rate | >= 0.30 | >= 0.35 |
| band coverage_abs_error | <= 0.15 | <= 0.08 |
| band_width_ic | > 0 | >= 0.05 |
| downside_width_ic | > 0 | >= 0.05 |
| width_bucket_realized_vol_ratio | > 1.0 | >= 1.15 |

band baseline은 Bollinger return band, historical quantile band, constant-width band로 고정한다. 다음 CP에서 기존 후보 재채점 또는 baseline 구현을 할 때 `docs/cp52_metric_definition_schema.json`을 기준으로 한다.

### CP53-M 재채점 기준값

CP53-M은 새 학습을 하지 않고 기존 checkpoint를 CP52 지표판으로 forward-only 재평가했다. 다음 실험 기준은 아래처럼 정리한다.

| 항목 | 기준값 |
|---|---|
| line 기본 후보 | PatchTST h5, seq_len 252, patch_len 32, stride 16 |
| line 기준선 | PatchTST h5, seq_len 252, patch_len 16, stride 8 |
| risk-only line 보조 | PatchTST h5, seq_len 252, patch_len 16, stride 4 |
| band 생존 후보 | CNN-LSTM seq_len 60, q15/q85, lambda_band 2.0, direct, scalar width calibration |
| band 확장 후보 | 같은 설정의 188/200 ticker 확인 run |
| composite 표시 정책 | `risk_first_upper_buffer_1.10` |

CP52 기준으로 band는 empirical coverage만 보지 않는다. q15/q85는 nominal coverage 0.70이며, `coverage_abs_error`, `interval_score`, `band_width_ic`, `downside_width_ic`를 함께 본다. CP53 기준 CNN-LSTM `s60_q15_b2_direct_188`은 empirical coverage 0.8069, coverage_abs_error 0.1069, band_width_ic 0.2463, downside_width_ic 0.0333이므로 후보가 아니라 생존 상태다.

다음 학습 또는 sweep 전에 baseline band 구현이 우선이다. Bollinger return band, historical quantile band, constant-width band를 같은 CP52 metric set으로 평가한 뒤 CNN-LSTM band sweep을 재개한다.

### CP54-M baseline 비교 기준값

CP54-M은 학습 없이 baseline만 같은 CP52 metric set으로 계산했다. 기본 평가 조건은 아래와 같다.

| 항목 | 값 |
|---|---|
| timeframe | 1D |
| horizon | 5 |
| seq_len | 252 |
| limit_tickers | 50 |
| target space | raw_future_return |
| q_low/q_high | 0.15 / 0.85 |
| nominal coverage | 0.70 |

Line 기준선 결과상 PatchTST h5 longer-context는 유지한다. test 기준 ic_mean 0.0241, long_short_spread 0.0034로 shuffled score IC 0.0213을 근소하게 넘고, false_safe_tail_rate 0.1475로 단순 기준선보다 보수성이 좋다.

Band 기준선 결과상 CNN-LSTM band는 아직 baseline을 이기지 못했다. `s60_q15_b2_direct_188`은 interval_score 0.1487, coverage_abs_error 0.1069였고, rolling historical quantile w252는 interval_score 0.1310, Bollinger return w60 k1은 coverage_abs_error가 거의 0이었다. 다음 band 실험은 `coverage` 단독이 아니라 `coverage_abs_error`, `interval_score`, `band_width_ic`, `downside_width_ic`를 baseline 대비로 판단한다.
