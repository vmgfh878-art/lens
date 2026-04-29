# Lens 모델 아키텍처 설계 문서

> **역할**: PatchTST·CNN-LSTM·TiDE 세 모델의 현재 구현, 표준 설계 갭, 개선 옵션을 한 곳에 모아 결정 가능한 형태로 제공.
> **사용 방법**: 각 섹션 마지막의 "결정 필요" 항목을 사용자가 골라주면, 해당 결정에 맞춰 CP3.5 (구조 보강) 지시서를 작성한다.
> **연관 문서**:
> - 학습 하이퍼파라미터: [`docs/training_hyperparameters.md`](training_hyperparameters.md)
> - 진행 기록: [`docs/project_journal.md`](project_journal.md)
> **작성 시점**: 2026-04-25 (CP3 직후, CP4 진입 전 구조 보강 단계).
> **마지막 업데이트**: 2026-04-27 (CP12 closure 기준 통합관리 보강).

---

## 0. 왜 이 문서가 필요한가

CP1~3는 데이터 파이프라인·sufficiency·split을 닫는 데 집중했고, 모델 구현은 "돌아가는 프로토타입" 수준에 머물렀다. 리뷰어 라운드에서 다음 갭이 드러났다:

1. **PatchTST에 RevIN 없음** — 비정상 시계열 정규화가 빠져 있어 종목별 스케일 변화에 약하다.
2. **PatchTST에 channel independence 없음** — 모든 피처를 한 패치로 묶어 서로 다른 스케일을 강제 공유한다.
3. **TiDE가 사실상 일반 MLP** — 인코더/디코더 분리, residual block, temporal decoder가 전부 없다.
4. **CNN-LSTM에 안정화 장치 없음** — LayerNorm, residual connection, attention pooling, gradient clipping이 빠져 있어 긴 시퀀스에서 학습이 흔들릴 가능성이 높다.
5. **출력 헤드 의미 혼란** — line head를 q50처럼 정렬에 섞어서 의미가 깨졌다 (P1-1).
6. **밴드 폭 손실 부호 오류** — `WidthPenaltyLoss`가 음수 폭을 보상한다 (P2-3).

이 문서의 목적은 사용자가 각 갭에 대해 "고친다 / 미룬다 / 다른 방향" 중 선택할 수 있게 옵션을 정리하는 것이다.

---

## 1. 시계열 모델 품질 평가 프레임워크 (8축)

각 모델을 평가할 때 다음 8축으로 본다.

| 축 | 질문 | 안 지키면 |
|---|---|---|
| **A. 비정상성 처리** | 종목·시기별 스케일·평균 이동을 모델이 어떻게 흡수하는가? | 종목 간 transfer 실패, 시기 변화에 취약 |
| **B. 채널 (피처) 독립성** | 17개 피처가 서로 다른 스케일·의미를 가지는데 어떻게 분리되는가? | 큰 스케일 피처가 작은 스케일 피처를 가림 |
| **C. 시간 순서 보존** | 시퀀스 위치 정보가 모델에 어떻게 들어가는가? | bag-of-bars처럼 동작, 추세/시점 정보 상실 |
| **D. 장기 의존성** | 멀리 떨어진 시점 (예: 60일 전) 정보가 현재 예측에 어떻게 반영되는가? | seq_len=252인데 마지막 20일만 영향 |
| **E. 정규화·안정화** | LayerNorm/BatchNorm/weight init이 학습 안정성을 어떻게 보장하는가? | Loss 발산, gradient 폭발 |
| **F. 출력 head 분리** | line·band·다른 quantile 출력이 서로 간섭 없이 학습되는가? | 한 head가 다른 head를 끌어내림 |
| **G. 크기·속도** | 파라미터 수, 추론 latency, 학습 시간이 데이터 규모와 균형 맞는가? | overfit (너무 크면) 또는 underfit (너무 작으면) |
| **H. 논문 fidelity** | "PatchTST"·"TiDE" 같은 이름이 실제 핵심 구조를 따르는가? | 이름값 못함, ablation 의미 약함 |

이후 각 모델 분석은 이 8축을 기준으로 한다.

---

## 2. 출력 헤드 구조 (line + band 분리 정착)

### 2.1 현 구현 (`ai/models/common.py`)

```python
class MultiHeadForecastModel(nn.Module):
    def __init__(self, hidden_dim, horizon):
        self.line_head = nn.Linear(hidden_dim, horizon)        # 점 예측 1개
        self.band_head = nn.Linear(hidden_dim, horizon * 2)    # 밴드 2개 (q10, q90)
```

출력 의미:
- `line` — AsymmetricHuberLoss(α=1, β=2)로 학습되는 **보수적 점 예측**. q50이 아님.
- `lower_band` (= q10) — PinballLoss(q=0.1)로 학습.
- `upper_band` (= q90) — PinballLoss(q=0.9)로 학습.

**핵심 원칙 (Lens v3 §3.4.2 확정)**: line head는 항상 별도 출력, 정렬 등 후처리에서 q50 자리에 끼워 넣지 않는다.

### 2.2 현재 결함

1. **추론 정렬이 line을 q50처럼 취급** (`ai/inference.py`:87~94)
   - `torch.sort([line, q10, q90])` 후 가운데 값을 line으로 재지정.
   - line의 학습 신호 (β=2 비대칭)가 추론에서 사라짐.
   - **수정**: sort에서 line 제외. `torch.sort([q10, q90])`만 적용 (또는 아예 정렬 안 하고 출력 파라미터화로 해결).

2. **검증·추론 후처리 불일치** (`ai/train.py`:213~220 vs `ai/inference.py`:87~94)
   - val coverage는 raw, 저장 예측은 정렬 후. 같은 파이프라인을 거쳐야 함.
   - **수정**: `ai/postprocess.py`에 `apply_band_postprocess(line, q_low, q_high)` 단일 함수, val·test·inference 모두 호출.

3. **밴드 폭 음수 가능** (`ai/loss.py`:73~84)
   - `mean(upper - lower)` 그대로. lower>upper일 때 음수.
   - **수정**: `mean(F.relu(upper - lower))` 또는 출력 파라미터화 (아래 옵션 참조).

### 2.3 출력 파라미터화 옵션 (밴드 crossing 구조적 제거)

| 옵션 | 구조 | 장점 | 단점 |
|---|---|---|---|
| **(현재) 직접 q10/q90 출력** | `band_head: hidden → 2` | 단순. 직관적. | crossing 후처리 필요. width 손실 부호 깨짐. |
| **(권장) 중심 + 폭 파라미터화** | `band_head: hidden → 2 = (center, log_half_width)`<br>`q10 = center - exp(log_half_width)`<br>`q90 = center + exp(log_half_width)` | 구조적으로 q10 ≤ q90 보장. width는 항상 양수. cross penalty·정렬 불필요. | center를 q50로 해석 가능 (line head와 충돌 가능성 — 별도 head 유지로 회피). |
| **단조 증가 파라미터화** | `lower = base, upper = base + softplus(δ)` | crossing 0 보장. | center 의미 흐릿. width 단독 의미 약함. |

**권고**: **중심 + log_half_width 파라미터화**. 이유:
- crossing 보장이 후처리(정렬)가 아닌 구조에서 나옴 → 학습 grad가 밴드 의미와 일치하게 흐름.
- width 손실은 `mean(2 * exp(log_half_width))`로 자연스럽게 양수.
- line head는 별개로 유지 → β=2 비대칭 그대로.
- cross penalty 손실 항 자체가 불필요해짐 → loss 단순화.

### 2.4 결정 필요 #A (출력 헤드 구조)

- [ ] **A-1**: 현재 직접 q10/q90 출력 유지 + sort/relu 후처리만 보완 (P1-1, P1-2, P2-3 fix).
- [ ] **A-2**: 중심 + log_half_width 파라미터화로 전환 (구조적 해결, 권장).
- [ ] **A-3**: 둘 다 구현하고 ablation으로 비교 (Phase 1 후반 연구 항목으로 격상).

---

## 3. PatchTST 분석

### 3.1 표준 구조 (Nie et al., 2023, "A Time Series is Worth 64 Words")

PatchTST의 핵심 4요소:

1. **Channel independence**: 입력 [B, L, C] (batch, seq, channel)을 [B*C, L, 1]로 reshape. **각 채널을 별도 시계열로 보고 동일 가중치 공유**. 멀티변량 conflation 회피.
2. **Patching**: 길이 L 시퀀스를 patch_len 단위로 나눠 N개 패치 토큰으로 만듦. Transformer가 N개 토큰만 처리 → 효율 ↑.
3. **RevIN (Reversible Instance Normalization)**: forward 시작에서 per-instance, per-channel 정규화. 출력 끝에서 역변환. **비정상성에 강건**.
4. **Transformer encoder + flatten head**: 인코딩된 N×d_model을 flatten해서 MLP head로 horizon 길이 예측.

### 3.2 현 구현 (`ai/models/patchtst.py`) — 갭 분석

| 축 | 표준 | 현재 | 평가 |
|---|---|---|---|
| A. 비정상성 (RevIN) | per-instance per-channel 정규화 | **없음** | ❌ 큰 갭 |
| B. 채널 독립성 | [B*C, L, 1] reshape, 가중치 공유 | 모든 채널을 한 패치로 flatten | ❌ 큰 갭 |
| C. 위치 정보 | learned positional embedding | learned positional embedding | ✓ |
| D. 장기 의존성 | Transformer self-attention | Transformer self-attention | ✓ |
| E. 정규화 | LayerNorm pre/post | LayerNorm post | ⚠ 부분 |
| F. head 분리 | line/band 별개 head | line/band 별개 head | ✓ |
| G. 크기 | d_model=128 (설정에 따라) | d_model=128 | ✓ |
| H. fidelity | — | 50% 수준 | ❌ |

### 3.3 개선 옵션

#### 옵션 PT-1: RevIN만 추가 (최소 보강)

**구현 비용**: 30~50줄. 0.5~1시간.

```python
class RevIN(nn.Module):
    def __init__(self, n_features, eps=1e-5, affine=True):
        ...
    def normalize(self, x):  # x: [B, L, C]
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + self.eps
        return (x - self.mean) / self.std * self.gamma + self.beta
    def denormalize(self, y):  # y: [B, H, ...]
        return (y - self.beta) / self.gamma * self.std + self.mean
```

forward 시작에서 normalize, head 출력 후 denormalize.

**장점**: 즉시 적용 가능. 비정상성 처리. README·hyperparameters 문서의 "RevIN" 명시와 코드 일치.
**단점**: channel independence 갭은 남음. PatchTST fidelity 60% 수준.

#### 옵션 PT-2: RevIN + Channel Independence 둘 다 (논문 fidelity 80%)

**구현 비용**: 100~150줄. 2~3시간.

핵심 구조:
```python
def forward(self, x):  # x: [B, L, C]
    B, L, C = x.shape
    x = self.revin.normalize(x)  # [B, L, C]
    x = x.permute(0, 2, 1)        # [B, C, L]
    x = x.reshape(B * C, L, 1)    # [B*C, L, 1]
    patches = self.patchify(x)     # [B*C, N, patch_len]
    z = self.patch_proj(patches)   # [B*C, N, d_model]
    z = z + self.pos_emb
    z = self.encoder(z)            # [B*C, N, d_model]
    z = z.reshape(B, C, -1)        # [B, C, N*d_model]
    z = self.flatten_head(z)       # [B, C, horizon]
    z = z.mean(dim=1)              # [B, horizon] — 채널 평균 (또는 attention pool)
    out = self.revin.denormalize(z)
    return self.build_output(out)
```

**장점**: 진짜 PatchTST. ablation 시 "PatchTST 효과" 주장 가능. 종목별 transfer에도 강함.
**단점**: 메모리 ×C배 증가 (17배). 학습 시간 늘어남. 본 학습 시 batch size 조정 필요.

#### 옵션 PT-3: PatchTST를 포기하고 다른 Transformer 변종

(예: Crossformer, Informer, iTransformer)

**평가**: Phase 1 범위에서 모델 종류를 갈아엎는 건 비용 대비 이득 약함. 기각.

### 3.4 결정 필요 #B (PatchTST 보강 범위)

- [ ] **B-1**: PT-1 (RevIN만). 빠른 보강, fidelity 60%.
- [ ] **B-2**: PT-2 (RevIN + Channel Independence). 진짜 PatchTST, fidelity 80%, 비용 2~3시간.
- [ ] **B-3**: 단계 진행 — 우선 PT-1, sweep 결과 보고 PT-2 결정.

---

## 4. CNN-LSTM 분석

### 4.1 표준 구조 (개념)

CNN-LSTM은 단일 논문 표준이 아니라 패턴이다. 일반적 권장:

1. **Conv 블록**: 1D Conv 2~3층 + BatchNorm/LayerNorm + ReLU/GELU + (옵션) residual.
2. **Sequence reduction**: Conv 출력을 시간 축으로 LSTM에 투입.
3. **LSTM**: 1~2층, hidden 보통 128~256.
4. **Pooling**: 마지막 hidden state만 사용 vs attention pooling.
5. **Regularization**: dropout, weight decay, gradient clipping.

### 4.2 현 구현 (`ai/models/cnn_lstm.py`) — 갭 분석

| 축 | 표준 | 현재 | 평가 |
|---|---|---|---|
| A. 비정상성 | per-instance norm 또는 입력 정규화 | 입력 정규화 (외부에서) | ⚠ 외부 의존 |
| B. 채널 독립성 | (CNN-LSTM 표준 아님) | 채널 mix (Conv1d가 모든 피처 섞음) | (해당 없음) |
| C. 위치 정보 | LSTM 자체가 순서 처리 | LSTM 사용 | ✓ |
| D. 장기 의존성 | LSTM (BPTT 한계 있음) | LSTM 2층 | ⚠ seq_len=252는 LSTM에 길음 |
| E. 정규화·안정화 | LayerNorm/BatchNorm, residual, grad clip | **없음** | ❌ 큰 갭 |
| F. head 분리 | — | line/band 별개 | ✓ |
| G. 크기 | hidden 128 적정 | hidden 128 | ✓ |
| H. fidelity | (CNN-LSTM은 자유로운 패턴) | 합리적 | ✓ |

### 4.3 개선 옵션

#### 옵션 CL-1: 안정화 장치만 추가 (보수적)

추가 사항:
1. Conv 블록에 LayerNorm (Conv 후, ReLU 전).
2. LSTM 출력에 LayerNorm.
3. Conv 블록에 residual connection (입력 차원 맞추기 위해 1x1 Conv 필요).
4. Gradient clipping (이미 학습 hyperparameter에 max_norm=1.0 결정).

**구현 비용**: 30~50줄, 0.5~1시간.
**장점**: 학습 안정성 ↑. 코드 변경 최소.
**단점**: 구조 한계는 그대로. seq_len=252는 LSTM에 부담.

#### 옵션 CL-2: Attention pooling 추가

LSTM 마지막 hidden 대신 모든 timestep을 attention으로 가중 평균.

```python
# scores: [B, L, 1]
scores = torch.softmax(self.attn(lstm_out), dim=1)
hidden = (lstm_out * scores).sum(dim=1)
```

**구현 비용**: 추가 20줄.
**장점**: 장기 의존성에서 멀리 떨어진 정보도 직접 참조. 해석 가능 (attention map).
**단점**: 살짝 무거워짐.

#### 옵션 CL-3: CNN-Transformer로 전환

LSTM 자체를 Transformer encoder로 교체. CNN이 local feature 추출, Transformer가 long-range.

**평가**: 사실상 PatchTST의 변종이 됨. 모델 다양성 측면에서 이 방향은 추천 안 함. 기각.

### 4.4 결정 필요 #C (CNN-LSTM 보강 범위)

- [ ] **C-1**: CL-1 (안정화 장치만 추가). 가장 보수적.
- [ ] **C-2**: CL-1 + CL-2 (안정화 + attention pooling). 권고.
- [ ] **C-3**: 현 상태 유지. (Phase 2 이월)

---

## 5. TiDE 분석

### 5.1 표준 구조 (Das et al., 2023, "Long-term Forecasting with TiDE")

TiDE의 핵심 5요소:

1. **Feature projection**: 각 시점 covariate를 작은 차원으로 projection (희소성 정리).
2. **Dense Encoder**: 과거 시퀀스 + projected covariate를 ResNet-style Dense Block 스택으로 인코딩. **MLP인데 residual connection이 핵심**.
3. **Dense Decoder**: 인코더 출력을 미래 horizon 길이로 변환.
4. **Temporal Decoder**: 디코더 출력 + 미래 covariate (있다면)를 timestep별로 다시 한 번 처리. 시점별 패턴 미세조정.
5. **Lookback skip**: 과거 시퀀스의 마지막 값을 출력에 직접 더함 (residual). 트렌드 baseline 역할.

### 5.2 현 구현 (`ai/models/tide.py`) — 갭 분석

| 축 | 표준 | 현재 | 평가 |
|---|---|---|---|
| A. 비정상성 | per-instance norm (논문은 RevIN 옵션) | 없음 | ❌ |
| B. 채널 독립성 | (TiDE는 multivariate 직접 처리) | flatten | ✓ |
| C. 위치 정보 | flatten 시 시간 축이 차원에 녹음 | flatten | ⚠ 약함 |
| D. 장기 의존성 | MLP는 모든 입력 동시 참조 | MLP 동시 참조 | ✓ |
| E. 정규화·안정화 | Residual block, LayerNorm, dropout | dropout만 | ❌ |
| F. head 분리 | — | line/band 별개 | ✓ |
| G. 크기 | hidden 256 적정 | hidden 256 | ✓ |
| H. fidelity | encoder-decoder, residual, temporal decoder | **그냥 4층 MLP** | ❌ 매우 큼 |

**현재 코드 본질**: `nn.Linear → ReLU → Dropout` 4번 반복한 일반 MLP. TiDE라고 부를 근거가 없다.

### 5.3 개선 옵션

#### 옵션 TD-1: 최소 TiDE 골격 (논문 fidelity 70%)

핵심 구성:

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(x + h)  # residual + norm

class TiDE(MultiHeadForecastModel):
    def __init__(self, ...):
        self.feature_proj = nn.Linear(n_features, feature_dim)  # per-timestep projection
        self.encoder = nn.Sequential(*[ResidualBlock(enc_dim) for _ in range(n_enc_layers)])
        self.decoder = nn.Sequential(*[ResidualBlock(dec_dim) for _ in range(n_dec_layers)])
        self.temporal_decoder = nn.Linear(dec_dim_per_step, hidden_dim)
        self.lookback_skip = nn.Linear(seq_len, horizon)  # 과거 → 미래 trend baseline
    def forward(self, x):  # x: [B, L, C]
        # 1. Feature projection
        x_proj = self.feature_proj(x)             # [B, L, F]
        # 2. Flatten + encoder
        h = x_proj.flatten(start_dim=1)            # [B, L*F]
        h = self.encoder(h)                        # [B, enc_dim]
        # 3. Decoder
        h = self.decoder(h)                        # [B, dec_dim]
        # 4. Temporal decoder (per-horizon-step)
        h = h.view(B, horizon, -1)                 # [B, H, dec_dim/H]
        h = self.temporal_decoder(h)               # [B, H, hidden_dim]
        # 5. Lookback skip
        skip = self.lookback_skip(x[..., 0])       # 첫 채널 (예: log_return)을 baseline으로
        # 6. build_output
        out = self.build_output(h.mean(dim=1)) + skip
        return out
```

**구현 비용**: 150~200줄, 2~3시간.
**장점**: TiDE 이름값. ablation에서 "encoder-decoder + residual의 효과" 주장 가능.
**단점**: 논문보다는 단순화. 미래 covariate 처리는 빠짐 (Lens는 예측 시점에 미래 covariate가 없으므로 OK).

#### 옵션 TD-2: 풀 TiDE 구현 (논문 fidelity 95%)

미래 covariate 처리, dropout schedule, layer count tuning 다 포함.

**구현 비용**: 300+줄, 5~7시간.
**평가**: 우리 데이터에 미래 covariate가 거의 없음 (시점 t에 t+h의 거시지표는 미지). 풀 구현의 추가 가치 작음. 기각.

#### 옵션 TD-3: 다른 MLP 계열로 교체

(예: N-BEATS, N-HiTS)

**평가**: 또 다른 모델 갈아엎기. Phase 1 범위 밖. 기각.

### 5.4 결정 필요 #D (TiDE 보강 범위)

- [ ] **D-1**: TD-1 (최소 TiDE 골격, fidelity 70%). 권고.
- [ ] **D-2**: TD-2 (풀 TiDE, fidelity 95%). 비용 대비 이득 작음.
- [ ] **D-3**: 현 상태 유지. (← 사용자 명시 거부)

---

## 6. Cross-cutting (모든 모델 공통)

### 6.1 Weight initialization

| 옵션 | 적용 | 비고 |
|---|---|---|
| **PyTorch 기본** | nn.Linear는 Kaiming uniform | 현재 사용 중. Transformer엔 보통 OK. |
| **Xavier/Glorot** | nn.init.xavier_uniform_ | LSTM·MLP에 권장되는 경우 있음. |
| **Truncated normal** | std=0.02 (BERT 류) | Transformer 표준. position embedding에 이미 적용. |

**권고**: PatchTST/CNN-LSTM/TiDE에 BERT 식 truncated normal init (std=0.02). 모델 비교 시 init 차이로 노이즈 끼는 것 방지.

### 6.2 Dropout 위치

| 모델 | 현재 | 권고 |
|---|---|---|
| PatchTST | TransformerEncoderLayer 내부 | + position embedding 직후 추가 |
| CNN-LSTM | LSTM 사이, 마지막 hidden | + Conv 후 추가 |
| TiDE | 매 layer 후 | residual block 내부에 통합 |

### 6.3 Loss 항 가중치 (`ForecastCompositeLoss`)

현재 (training_hyperparameters.md 참조):
- `λ_line = 1.0` (AsymmetricHuber)
- `λ_band = 1.0` (Pinball, q10+q90 합)
- `λ_width = 0.05` (밴드 폭 페널티)
- `λ_cross = 1.0` (crossing 페널티)

**문제**: P2-3 — width loss가 음수 가능. 출력 파라미터화 (옵션 A-2) 채택 시 width는 항상 양수, cross loss는 불필요해짐 → λ_cross 삭제, `λ_width = 0.05` 그대로.

**스윕 대상**: λ_band, λ_width, λ_line의 비율은 A-1 sweep (주차 2 후반)에서 결정.

### 6.4 결정 필요 #E (Cross-cutting)

- [ ] **E-1**: BERT 식 init (std=0.02) 전 모델 적용. (권고)
- [ ] **E-2**: Dropout 위치 권고대로 보강. (권고)
- [ ] **E-3**: Loss 구성은 출력 파라미터화 (A-2) 결정 후 따라감. (의존성)

---

## 7. 결정 필요 항목 요약 (확정 — 2026-04-25)

| 코드 | 항목 | 확정 옵션 | 비고 |
|---|---|---|---|
| **A** | 출력 헤드 구조 | **A-3** | 두 방식 모두 구현 + ablation |
| **B** | PatchTST 보강 | **B-2** | RevIN + Channel Independence (논문 fidelity 80%) |
| **C** | CNN-LSTM 보강 | **C-2** | LayerNorm + residual + grad clip + attention pooling |
| **D** | TiDE 보강 | **D-1** | 최소 골격 (논문 fidelity 70%). 미래 covariate 처리 생략 (Lens 데이터에 미래 covariate 없음) |
| **E1** | Init 통일 | **적용** | BERT 식 trunc_normal(std=0.02) 전 모델 |
| **E2** | Dropout 위치 보강 | **적용** | 모델별 표준 위치 보강. 비율은 sweep |

**시나리오 매핑**: 본 결정은 §8 "최대 fidelity" 시나리오에 해당. 예상 비용 13~18시간.

다음 단계: 위 결정 기반으로 CP3.5 지시서 작성. P1-1·P1-2·P2-3 fix와 함께 한 묶음으로 처리.

---

## 8. 비용·시간 견적 (옵션 조합별)

| 시나리오 | 포함 | 예상 시간 | 결과물 |
|---|---|---|---|
| **최소 fix** | A-1 + B-1 + C-1 + D-1 + E1 + E2 + P1-1·P1-2·P2-3 | 6~9시간 | 모든 모델 이름값, 출력 정합성 회복 |
| **권고 (균형)** | A-2 + B-2 + C-2 + D-1 + E1 + E2 + P1-1·P1-2·P2-3 | 10~14시간 | PatchTST·TiDE fidelity 80%, CNN-LSTM 안정화, 출력 구조적 정합 |
| **최대 fidelity** | A-3 + B-2 + C-2 + D-1 + E1 + E2 + P1-1·P1-2·P2-3 | 13~18시간 | 출력 두 방식 ablation까지 |

**권고: "균형"** 시나리오. 학부 6주 일정 내에서 fidelity와 비용 trade-off가 가장 합리적.

---

## 9. CHANGELOG

- **2026-04-25 (초판)**: CP3 종료 후 리뷰어 라운드에서 P1-1, P1-2, P2-1, P2-2, P2-3 5건 도출. 사용자가 모델 아키텍처 갭 정리·옵션 비교 요청. 본 문서 작성. 결정 필요 항목 6개 (A, B, C, D, E1, E2) 표시.
- **2026-04-25 (결정 확정)**: 사용자 직접 선택 — A-3 / B-2 / C-2 / D-1 / E-1 적용 / E-2 적용. "비용보다 fidelity 우선" 원칙 명시. TiDE D-1 채택은 미래 covariate 처리가 Lens 데이터에 불필요하다는 합의에 따른 의도된 갭. CP3.5 지시서 작성 단계 진입.

---

## 10. 2026-04-27 통합관리 업데이트 (CP4~CP12)

이 섹션은 CP3.5 이후 구현 에이전트가 반영한 구조 변경과 오케스트레이터 검수 결론을 한 번에 묶은 최신 기준점이다. 위 본문은 당시 의사결정 기록으로 유지하고, 실제 다음 CP 발주는 아래 기준을 우선한다.

### 10.1 Lens 정체성 잠금

아래 두 가지는 성능 개선 과정에서도 바꾸지 않는다.

1. **예측선은 하방 보수적 예측선이다.**
   - line head는 q50 평균선이 아니다.
   - `AsymmetricHuberLoss`의 α=1, β=2 철학을 유지한다.
   - 과대예측을 더 위험하게 보고, 상승을 과하게 말하는 모델보다 덜 낙관적인 모델을 우선한다.

2. **밴드는 Lens의 본체다.**
   - lower/upper band는 보조 장식이 아니라 AI가 구축하는 리스크 구간이다.
   - coverage, avg_band_width, band_loss, calibration은 모델 채택의 핵심 guardrail이다.
   - direction, rank, IC, long-short spread는 성능 개선을 위한 보조 신호이며, 밴드 품질을 훼손하면서 채택하지 않는다.

허용되는 변경 범위는 넓다. target 재설계, direction/rank 보조 head, ranker, DLinear/NLinear, iTransformer, NHITS, LightGBM/CatBoost 비교는 모두 가능하다. 다만 결과 해석은 항상 "하방 보수적 line + calibrated AI band" 위에서 해야 한다.

### 10.2 현재 공통 출력 계약

`ForecastOutput`의 최신 계약은 다음과 같다.

| 출력 | 상태 | 의미 |
|---|---|---|
| `line` | 필수 | 하방 보수적 예측선 |
| `lower_band` | 필수 | 하방 quantile 리스크 구간 |
| `upper_band` | 필수 | 상방 quantile 리스크 구간 |
| `direction_logit` | 선택 | CP11에서 추가된 방향 보조 head 출력. 기본값 `None` |

direction head는 현재 CNN-LSTM에만 옵션으로 붙어 있다. PatchTST와 TiDE에는 아직 미러링하지 않았다. CP13 후보지만, CP12 CUDA/bf16 NaN 매트릭스가 통과하기 전까지 확장하지 않는다.

### 10.3 모델별 최신 구조

| 모델 | 최신 반영 | 현재 판단 |
|---|---|---|
| PatchTST | RevIN denorm 보강, Channel Independence aggregate, ticker embedding, precomputed bundle 기반 학습 경로 | 논문 fidelity와 Lens head 구조를 함께 유지하는 주력 후보. 긴 sweep보다 target/loss/eval 재설계 후 재평가 |
| CNN-LSTM | 4-layer dilated TCN `[1,2,4,8]`, receptive field 31, attention/LSTM 안정화, 선택적 direction head | 빠른 구조 실험용 모델. CP11 direction head는 50티커 기준 이득 불확실. CUDA/bf16 NaN 원인 분리 후 재평가 |
| TiDE | per-step head, horizon decoder, calendar future covariate 7채널 주입 | CP3.5의 "미래 covariate 생략" 판단은 거시·재무 미지 변수에는 맞았지만 캘린더 known future covariate에는 불완전했다. CP6.5에서 정정 완료 |

### 10.4 CP11 direction head 판정

CNN-LSTM direction head는 구조적으로는 붙었지만 채택 확정은 아니다.

- 5티커 1 epoch smoke에서는 direction_accuracy, IC, long-short spread가 좋아 보였다.
- 50티커 짧은 검증에서는 coverage와 band_loss만 소폭 개선됐고, avg_band_width, overprediction_rate, mean_overprediction, direction_accuracy, long-short spread가 악화됐다.
- 따라서 "direction head가 확실히 이득"이라고 말하면 틀리다.

다음 판단 기준은 direction_accuracy 단독이 아니라 다음 묶음이다.

- band_loss 유지 또는 개선
- coverage 0.75~0.85 근처 유지
- avg_band_width 과도 확대 금지
- overprediction_rate와 mean_overprediction 악화 금지
- Spearman IC, top-k spread, fee-adjusted return 동시 확인

### 10.5 CP12 안정성 계층

CP12는 모델 구조를 키운 CP가 아니라, NaN 결과가 실험판을 오염시키지 못하게 막은 신뢰성 CP다.

- `ai/finite.py`에 finite gate 유틸 추가.
- train, validation, test, checkpoint 저장 전 4단 finite gate 적용.
- NaN run은 `model_runs.status="failed_nan"`로 메타만 남기고, predictions, prediction_evaluations, backtest_results, checkpoint 저장은 차단.
- inference/backtest는 `status != "completed"` run을 거부한다.
- `--amp-dtype {bf16, fp16, off}`가 추가됐고, 기본값은 기존과 같은 `bf16`이다.

이 변경은 line/band 손실, target plumbing, 모델 head 철학을 바꾸지 않는다. CP11에서 발견한 CNN-LSTM + CUDA + bf16 NaN을 재현 가능하게 추적하기 위한 안전장치다.

### 10.6 다음 구조 발주 기준

CP12 매트릭스 결과에 따라 구조 발주를 분기한다.

| 조건 | 다음 발주 |
|---|---|
| CUDA amp off는 PASS, bf16만 NaN | CP12.5: `--fp32-modules lstm,heads` 같은 부분 fp32 강제 옵션 검토 |
| 50티커 bf16 baseline까지 PASS | CP13: `lambda_direction` sweep, PatchTST/TiDE direction head 미러링으로 fairness 비교 |
| direction head만 NaN | CP11 direction_logit/BCEWithLogits 경로 단독 분리 |
| baseline도 NaN | direction 실험 금지. 입력, precision, LSTM/Conv 경로부터 분리 |

### 10.7 CP12 매트릭스 결과 반영

사용자 GPU에서 CP12 매트릭스 6개를 실행한 결과는 다음과 같다.

| # | 조건 | 결과 | 핵심 신호 |
|---|---|---|---|
| 1 | 5티커 CPU fp32 baseline | PASS | NaN 없음 |
| 2 | 5티커 CUDA amp off baseline | PASS | 결과 JSON 정상. 단, 종료코드 1 |
| 3 | 5티커 CUDA bf16 baseline | NAN | val `avg_band_width=nan` |
| 4 | 5티커 CUDA amp off direction_head | PASS | 결과 JSON 정상. 단, 종료코드 1 |
| 5 | 5티커 CUDA bf16 direction_head | NAN | val `avg_band_width=nan` |
| 6 | 50티커 CUDA bf16 baseline | NAN | val `avg_band_width=nan` |

판정: direction head 단독 문제가 아니다. CNN-LSTM baseline도 CUDA bf16 autocast에서 val metric NaN을 만든다. 따라서 CP13으로 넘어가면 안 된다. 다음 발주는 CP12.5이며, 목표는 **CNN-LSTM CUDA bf16 안정화와 NaN 위치 진단 강화**다.

추가 관찰: `[2]`, `[4]`는 출력상 PASS지만 프로세스 종료코드가 1이다. 이 문제는 NaN과 별개로 정상 종료 경로에 숨은 예외나 cleanup 오류가 있을 수 있으므로 CP12.5에 함께 포함한다.

### 10.8 CP12.5 결과 반영

CP12.5 매트릭스 결과, CNN-LSTM CUDA bf16 NaN은 LSTM 구간 문제로 좁혀졌다.

| 케이스 | 조건 | 결과 | 판단 |
|---|---|---|---|
| A | bf16, `--fp32-modules none` | NAN | val aggregate `tensor:raw.line`부터 NaN. target은 finite |
| B | bf16, `--fp32-modules lstm` | PASS | 지표 정상. 단, 종료코드 `-1073740791` |
| C | bf16, `--fp32-modules heads` | NAN | heads만 fp32로는 해결 불가 |
| D | bf16, `--fp32-modules lstm,heads` | PASS | baseline 지표 정상. 단, 종료코드 `-1073740791` |
| E | bf16, direction_head, `--fp32-modules lstm,heads` | PASS | direction head 경로도 지표 정상. 단, 종료코드 `-1073740791` |
| F | 50티커 bf16 baseline, `--fp32-modules lstm,heads` | PASS | sweep 조건 규모에서도 지표 정상. 단, 종료코드 `-1073740791` |
| G | CUDA amp off baseline 종료코드 재확인 | 실패 | 출력은 정상이나 종료코드 `-1073740791` |

구조 판단:

- NaN의 1차 원인은 direction head가 아니라 CNN-LSTM LSTM bf16 autocast 경로다.
- `lstm`만 fp32로 강제해도 baseline NaN은 해결된다.
- `lstm,heads`는 baseline, direction head, 50티커 baseline까지 지표상 통과하므로 다음 안정화 후보값이다.
- 하지만 CUDA 성공 run의 프로세스 종료코드가 계속 비정상이므로 CP12.5는 닫지 않는다.

다음 발주:

- **CP12.6**: CUDA 성공 run 종료코드 `-1073740791` 원인 분리와 정상 종료 보장.
- CP13 direction/rank 실험은 여전히 보류한다.

### 10.9 CP12.6 closure와 Phase 1 범위 전환

CP12.6 결과, CUDA 성공 run 종료코드 문제는 train.py 종료부 일반 로직이 아니라 Windows CUDA 환경의 cuDNN LSTM 경로 종료 크래시로 분리됐다. `CNN-LSTM`의 LSTM 실행 구간에서 CUDA일 때 `torch.backends.cudnn.flags(enabled=False)`를 적용한 뒤 다음 조건이 모두 통과했다.

| 조건 | 결과 |
|---|---|
| 5티커 CPU amp off baseline | exit code 0 |
| 5티커 CUDA amp off baseline | exit code 0 |
| 5티커 CUDA amp off + explicit cleanup | exit code 0 |
| 5티커 CUDA bf16 + `--fp32-modules lstm,heads` | exit code 0 |
| 50티커 CUDA bf16 + `--fp32-modules lstm,heads` | exit code 0 |

따라서 CP12.6은 closure 처리한다. 단, 이 결과는 CNN-LSTM 연구를 Phase 1에서 계속한다는 뜻이 아니다. 오히려 런타임 안정성만 확보하고, 모델 연구 범위는 아래처럼 재정의한다.

| 단계 | 모델 범위 | 목적 |
|---|---|---|
| Phase 1 | **PatchTST 단일 주력** | Lens 제품 루프와 발표 가능한 end-to-end demo 완성 |
| Phase 1.5 | TiDE, CNN-LSTM 재도입 | PatchTST 기준선과 UI/API가 잡힌 뒤 비교 모델 확장 |
| Phase 2 | ranker, ensemble, 다른 universe | 연구 확장과 성능 고도화 |

이 전환은 후퇴가 아니다. 모델 3개를 동시에 최적화하면서 생긴 checkpoint-수정-버그 루프를 끊고, 하나의 신뢰 가능한 축을 먼저 완성하기 위한 범위 축소다.

### 10.10 PatchTST Solo Track 설계 원칙

PatchTST 원논문과 공식 구현에서 프로젝트에 직접 적용할 원칙은 다음이다.

| 원칙 | Lens 적용 |
|---|---|
| patching은 핵심이다 | `patch_len`, `stride`, `seq_len`을 lr보다 상위 sweep 축으로 올린다 |
| channel independence는 핵심이다 | Phase 1 기본값은 `channel_independent=True`, `ci_aggregate=target` 유지. `mean/attention`은 ablation으로만 비교 |
| 긴 lookback을 활용한다 | 1D는 252뿐 아니라 504 후보를 작게 검증한다. 단, 속도와 sufficiency 감소를 같이 기록 |
| self-supervised pretraining은 강한 옵션이다 | 바로 붙이지 않고 CP 후반 후보로 둔다. 먼저 supervised target/loss/eval을 안정화 |
| 모델 구조보다 평가 목적이 먼저다 | direction head 확대보다 `market_excess_return`, volatility-normalized return, band calibration을 먼저 비교 |

PatchTST Solo Track에서 당장 금지할 것:

- PatchTST, TiDE, CNN-LSTM에 같은 변경을 동시에 미러링하지 않는다.
- direction/rank head를 먼저 붙이지 않는다.
- 긴 sweep을 먼저 돌리지 않는다.
- band 품질을 훼손하면서 direction_accuracy만 올리는 실험은 채택하지 않는다.

### 10.11 CP13 closure

CP13 산출물 `docs/cp13_patchtst_solo_plan.md`를 기준으로 PatchTST Solo Track 범위를 닫는다.

확인된 내용:

- TiDE/CNN-LSTM freeze 범위가 Phase 1.5 backlog로 분리됐다.
- PatchTST 현재 구현 감사표가 작성됐다.
- 공식 구현 대비 의도적 차이와 실수 가능성이 있는 차이가 분리됐다.
- 실험 축 우선순위가 target type → patch geometry → seq_len → ci_aggregate → capacity 순서로 재정의됐다.
- 현재 `ai/train.py`와 `ai/sweep.py`에는 `patch_len`, `stride`, `d_model`, `n_layers`, `n_heads`가 CLI/sweep 축으로 아직 노출되어 있지 않다.

CP14는 바로 긴 sweep으로 가지 않는다. 먼저 `raw_future_return`과 `volatility_normalized_return` 2개 target baseline을 같은 조건에서 비교하고, 동시에 PatchTST 전용 실험 인자를 열지 여부를 결정한다.

### 10.12 제품 화면의 line/band 표시 계약

CP16-P 이후 제품 화면은 PatchTST의 저장된 prediction을 다음 방식으로 표시한다. 이는 모델 구조 변경이 아니라, Lens 정체성을 제품 화면에 드러내는 표시 계약이다.

| 화면 요소 | 사용 필드 | 의미 |
|---|---|---|
| 보수적 예측선 | `line_series` 우선, 없으면 `conservative_series` | 하방 보수적 line |
| AI 밴드 상단 | `upper_band_series` | 예측 리스크 구간 상단 |
| AI 밴드 하단 | `lower_band_series` | 예측 리스크 구간 하단 |
| 날짜 축 | `forecast_dates` | 예측 구간 날짜 |
| 밴드 분위수 | `band_quantile_low`, `band_quantile_high` | q_low/q_high 메타 정보 |

표시 원칙:

- 1D/1W만 AI overlay를 표시한다.
- 1M은 가격 전용이다.
- overlay series 길이와 `forecast_dates` 길이가 다르면 임의 보정하지 않고 표시하지 않는다.
- band fill은 보류하고 상하단 점선으로 먼저 표시한다.
- 제품 화면에서는 `raw_future_return`, `volatility_normalized_return` 같은 target 용어를 전면 노출하지 않는다.

### 10.13 제품 화면의 평가 지표 표시 계약

CP17-P 이후 제품 화면은 모델 구조를 바꾸지 않고, 저장된 run의 평가 지표를 read-only로 보여준다.

| 화면 | 기준 run | 표시 |
|---|---|---|
| 주식 보기 | latest completed PatchTST run | prediction overlay와 ticker/asof 평가 요약 |
| 백테스트 | 선택 timeframe의 latest completed PatchTST run | 백테스트 결과와 `val_metrics`/`test_metrics` 품질 요약 |
| 모델 학습 | 선택한 status/run | config, checkpoint, `val_metrics`/`test_metrics`, ticker/asof 평가 테이블 |

원칙:

- `failed_nan` run은 주식 보기/백테스트 결과에 섞지 않는다.
- 모델 학습 화면에서는 `failed_nan`도 상태 추적 목적으로 보여준다.
- 제품 화면 지표는 Research guardrail과 같은 의미를 유지한다.

### 10.14 PatchTST 입력 정합 복구 상태

PatchTST 학습 경로의 입력 feature finite contract는 복구됐다.

정리:

- 재무 6개 컬럼 `revenue`, `net_income`, `equity`, `eps`, `roe`, `debt_ratio`의 결측은 학습 tensor 진입 전 0.0으로 채운다.
- `has_fundamentals`는 원본 재무값 존재 여부를 나타내는 플래그로 유지한다.
- NaN이 모델 입력으로 들어가 loss가 즉시 NaN이 되던 CP14-R 실패 원인은 닫혔다.
- `raw_future_return` 전체 학습은 NaN 없이 완료됐다.
- `volatility_normalized_return`은 아직 50티커 smoke부터 다시 확인해야 한다.

이제 PatchTST solo track의 우선순위는 target 비교를 재개하되, 전체 run이 아니라 smoke-first 운영으로 바뀐다.

### 10.15 데모용 prediction run 선택 정책

CP19 이후 주식 보기 화면은 latest completed run만 고집하지 않는다. 제품 화면의 목적은 사용자에게 실제 표시 가능한 AI 밴드와 보수적 예측선을 보여주는 것이므로, 다음 정책을 따른다.

| 단계 | 동작 |
|---|---|
| 1 | 선택 timeframe의 completed PatchTST run을 최신순으로 조회 |
| 2 | 선택 ticker의 prediction row가 있는지 순서대로 확인 |
| 3 | `forecast_dates`, `line_series`, `upper_band_series`, `lower_band_series`가 유효한 첫 run을 사용 |
| 4 | 사용 가능한 prediction이 없으면 가격 차트와 empty state를 유지 |

현재 데모 기준:

- 최신 completed run `patchtst-1D-94d61c4e84d3`은 AAPL prediction이 없다.
- 사용 가능한 demo run은 `patchtst-1D-fc096a026a1e`이다.
- 이 정책은 fake data를 만들지 않고 실제 저장 산출물만 사용한다.
### CP21-R PatchTST 실험 인자 노출

PatchTST Solo Track에서 구조 자체는 유지하되, geometry 실험에 필요한 인자를 학습 CLI에서 직접 조절할 수 있게 했다.

노출된 인자:

| CLI | PatchTST 인자 | 기본값 |
|---|---|---:|
| `--patch-len` | `patch_len` | 16 |
| `--patch-stride` | `stride` | 8 |
| `--patchtst-d-model` | `d_model` | 128 |
| `--patchtst-n-heads` | `n_heads` | 8 |
| `--patchtst-n-layers` | `n_layers` | 3 |

적용 원칙:

- 위 인자는 `model=patchtst`일 때만 모델 생성에 전달한다.
- TiDE와 CNN-LSTM에는 영향을 주지 않는다.
- 기본값은 기존 PatchTST 기본 구현과 동일하게 유지한다.
- checkpoint/config summary에는 `TrainConfig`를 통해 값이 기록된다.

CP21-R smoke 결과, `patch_len=16`, `stride=8` baseline이 투자 지표 기준으로 가장 안정적이었다. `patch_len=32`, `stride=16`은 밴드 폭과 속도 측면의 보조 후보지만 주력 geometry로 채택하지 않는다.
### CP23-R PatchTST band calibration 상태

PatchTST 구조는 변경하지 않았다. CP23-R은 모델 architecture 변경이 아니라 q 범위와 `lambda_band`만 조절한 calibration smoke다.

현재 구조 기준:

- Patch geometry: `patch_len=16`, `stride=8`
- Channel independence: 유지
- `ci_aggregate=target`: 유지
- Band mode: `direct` 우선

현재 calibration 후보:

- 1차 후보: `q_low=0.30`, `q_high=0.70`, `lambda_band=2.0`
- 보수 후보: `q_low=0.25`, `q_high=0.75`, `lambda_band=2.0`

`q35-b2`는 coverage가 이상적 구간에 들어왔지만 upper breach가 커져 CP24-R의 Bollinger 기준선 비교 전까지는 보류했다.

### CP24-R 이후 PatchTST band 역할 분리

CP24-R에서 Bollinger return-space 기준선과 비교한 결과, PatchTST 구조 자체는 그대로 유지한다. 변경 대상은 architecture가 아니라 calibration preset이다.

현재 역할:

- 보수형: `q25-b2`. coverage가 높고 투자 지표 보존이 좋지만 Bollinger 대비 밴드 폭이 넓다.
- 기본형: `q30-b2`. BB20-2.0s와 BB20-1.5s 사이의 breach 위치에 있어 다음 안정성 확인의 1차 후보로 둔다.
- 공격형: `q35-b2`. upper breach가 BB20-1.5s와 비슷하고 BB20-1.0s보다 낮아 공격형 후보로 유지한다.

다음 검증은 구조 변경 없이 `q30-b2`, `q35-b2`를 100티커로 넓혀 band asymmetry와 투자 지표 안정성을 확인한다.

### CP25-R 이후 calibration 상태

100티커 안정성 smoke에서 `q30-b2`, `q35-b2`는 모두 탈락했다. 이는 PatchTST 구조 문제가 아니라 calibration preset이 50티커 subset에 의존했을 가능성이 큰 결과다.

구조 고정 사항:

- PatchTST geometry: `patch_len=16`, `stride=8` 유지
- channel independence: 유지
- `ci_aggregate=target`: 유지
- direction/rank head: 추가 금지 유지
- loss 구조: 변경 금지 유지

다음 작업은 architecture 변경이 아니라 reserve preset인 `q25-b2`와 필요 시 `q20-b2`를 100티커 기준으로 다시 검증하는 것이다.

### CP26-R 이후 PatchTST calibration 상태

CP26-R에서도 PatchTST 구조는 변경하지 않았다. 변경 없이 q25/q20/q15 calibration preset만 100티커에서 재확인했다.

구조 고정:

- `patch_len=16`, `stride=8`
- channel independence 유지
- `ci_aggregate=target`
- `band_mode=direct`
- direction/rank head 없음
- loss 구조 변경 없음

결과적으로 기본형 preset은 아직 확정하지 못했다. 최종 checkpoint 기준으로는 `q15-b2`만 공격 후보로 남았다.

다음 병목은 모델 구조가 아니라 checkpoint selection 정책이다. validation total loss 기준으로 best를 고르면 band coverage가 과도하게 희생될 수 있으므로, 다음 단계에서는 coverage-aware selection을 추가하거나 별도 selector로 checkpoint 후보를 골라야 한다.

### CP28-R 이후 PatchTST 생존 판정

CP27-R에서 coverage-aware checkpoint selection을 추가해 100티커 기준 q20/q25/q15가 후보권으로 복구됐지만, CP28-R의 200티커 안정성 확인에서는 세 preset 모두 실패했다. 모델 구조는 계속 고정했다.

구조 고정:

- `patch_len=16`, `stride=8`
- channel independence 유지
- `ci_aggregate=target`
- `band_mode=direct`
- direction/rank head 없음
- loss 구조 변경 없음

200티커 결과는 구조 개선보다 baseline 비교가 먼저 필요하다는 쪽으로 기운다. q20/q25는 coverage가 0.50~0.60대로 무너졌고, q15도 upper breach가 0.15를 넘었다. 따라서 현재 PatchTST preset은 full 473티커 후보가 아니며, 다음 단계에서는 DLinear/NLinear 같은 단순 시계열 baseline을 같은 evaluation contract로 붙여 비교해야 한다.
### CP29-D 이후 가격 피처 계약

PatchTST 구조는 CP29-D에서 바꾸지 않았다. 변경점은 모델이 먹는 가격 파생 피처의 입력 계약이다. 기존에는 raw EODHD `open/high/low`와 `adjusted_close` 기반 previous close가 섞일 수 있어 `open_ratio/high_ratio/low_ratio`가 비정상적으로 폭주했다.

이제 모델 입력 가격 피처는 adjusted OHLC 기준으로 고정한다. `adj_factor = adjusted_close / close`를 만든 뒤 `adj_open`, `adj_high`, `adj_low`, `adj_close`로 가격 파생 피처를 계산한다. PatchTST의 `n_features=36`, patch geometry, channel independence, `ci_aggregate=target`, `band_mode=direct`는 그대로 유지한다.

CP29-D smoke는 구조 성능 판정이 아니라 입력 계약 검증이다. 50티커 1epoch에서 feature finite gate는 통과했고, full 473티커 학습은 계속 금지한다.

### CP30-G 이후 실험 게이트 계약

CP30-G는 모델 구조를 바꾸지 않았다. PatchTST, RevIN, channel independence, patch geometry, head 구조는 그대로 두고, 저장/추론/백테스트 재현성 계약만 수리했다.

모델 run 저장 계약:

- `model_runs.band_mode`는 config JSON이 아니라 정식 컬럼에도 저장한다.
- `model_runs.feature_version`은 `FEATURE_CONTRACT_VERSION=v3_adjusted_ohlc`를 따른다.
- coverage gate가 eligible checkpoint를 찾지 못하고 fallback한 run은 `failed_quality_gate`로 저장한다.
- inference와 backtest는 기존 CP12 정책대로 `status="completed"` run만 허용하므로, quality gate 실패 run은 제품 산출물로 승격되지 않는다.

추론 재현성 계약:

- ticker embedding이 활성화된 checkpoint는 checkpoint config의 `ticker_registry_path`를 필수로 사용한다.
- inference subset이 새 ticker registry를 만들지 않는다.
- registry `timeframe`, `num_tickers`, mapping 크기가 checkpoint config와 다르면 실패한다.

백테스트 계약:

- price decode와 realized return anchor는 adjusted 기준이다.
- signal backtest는 row 단위 position shift가 아니라 날짜별 포트폴리오 단위로 계산한다.
- 같은 날짜의 `BUY/SELL` 활성 포지션은 절대 노출 합 1로 정규화하고, turnover는 이전 날짜 weight와 현재 날짜 weight의 L1 변화량으로 계산한다.

RevIN은 이번 CP에서 수정하지 않는다. 다음 ablation에서는 CP29/CP30 계약을 고정한 뒤 `use_revin=True/False`만 분리해 비교한다.
