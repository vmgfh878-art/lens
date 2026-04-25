# Lens 모델 아키텍처 설계 문서

> **역할**: PatchTST·CNN-LSTM·TiDE 세 모델의 현재 구현, 표준 설계 갭, 개선 옵션을 한 곳에 모아 결정 가능한 형태로 제공.
> **사용 방법**: 각 섹션 마지막의 "결정 필요" 항목을 사용자가 골라주면, 해당 결정에 맞춰 CP3.5 (구조 보강) 지시서를 작성한다.
> **연관 문서**:
> - 학습 하이퍼파라미터: [`docs/training_hyperparameters.md`](training_hyperparameters.md)
> - 진행 기록: [`docs/project_journal.md`](project_journal.md)
> **작성 시점**: 2026-04-25 (CP3 직후, CP4 진입 전 구조 보강 단계).

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
