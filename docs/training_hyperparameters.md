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
