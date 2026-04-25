# Lens 학습 하이퍼파라미터 & 설계 결정 (SoT)

> **목적**: Phase 1 모델 학습에 관한 모든 파라미터·기법·설계 결정을 한 파일로 묶는다. 결정값·근거·고려한 대안·탈락 이유·민감도·Phase 2 재검토 트리거를 모두 명시. 새 결정이 생기면 **이 파일을 Edit로 업데이트** (rewrite 금지, 섹션별 append).
>
> 마지막 업데이트: 2026-04-25
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
