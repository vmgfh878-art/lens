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

### CP32-M 이후 PatchTST 구조 판정

CP32-M에서는 PatchTST 구조를 전혀 변경하지 않고 `v3_adjusted_ohlc` clean feature 기준으로 기존 preset을 재검증했다.

고정 구조:

- `patch_len=16`, `stride=8`
- channel independence 유지
- `ci_aggregate=target`
- `band_mode=direct`
- direction/rank head 없음
- loss 구조 변경 없음

50티커 clean feature 재검증 결과, baseline/q20/q15는 coverage가 0.98~1.00에 가까워 밴드가 과도하게 넓었고 coverage gate fallback으로 떨어졌다. q25-b2만 coverage gate는 통과했지만 coverage 0.949298로 기본형 범위를 넘고 test fee가 음수였다.

따라서 현재 구조의 기존 preset은 full run 후보가 아니다. 이 결론은 PatchTST 구조 자체의 최종 폐기가 아니라, 기존 q preset과 현재 구조 조합이 clean feature 기준에서 유효하지 않다는 뜻이다.

### CP33-M 이후 RevIN 판정

CP33-M에서 PatchTST의 `use_revin`을 CLI로 노출하고 RevIN output denormalize 영향을 분리했다. 구조 자체는 유지했고, 새 head나 새 target은 추가하지 않았다.

결과적으로 `use_revin=False`는 밴드 문제를 해결하지 못했다. q25/q30/q35 모두 coverage가 0.96~0.99 수준으로 높았고, validation long-short spread가 음수였다. 이는 과보수 밴드 문제가 RevIN denormalize 하나로 설명되지 않는다는 뜻이다.

현재 구조 판정:

- PatchTST 기존 preset 실험은 중단한다.
- `use_revin` 옵션은 ablation과 checkpoint 재현성을 위해 유지한다.
- 다음 모델 계층 판단은 DLinear/NLinear baseline을 같은 clean feature/evaluation contract에 붙여 비교하는 방향이 맞다.
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
- 학습 registry는 ticker 집합 fingerprint가 있는 파일로 저장해 이후 run이 같은 timeframe 공용 파일을 덮어쓰지 못하게 한다.
- registry `timeframe`, `num_tickers`, mapping 크기가 checkpoint config와 다르면 실패한다.

백테스트 계약:

- price decode와 realized return anchor는 adjusted 기준이다.
- signal backtest는 row 단위 position shift가 아니라 날짜별 포트폴리오 단위로 계산한다.
- 같은 날짜의 `BUY/SELL` 활성 포지션은 절대 노출 합 1로 정규화하고, turnover는 이전 날짜 weight와 현재 날짜 weight의 L1 변화량으로 계산한다.

RevIN은 이번 CP에서 수정하지 않는다. 다음 ablation에서는 CP29/CP30 계약을 고정한 뒤 `use_revin=True/False`만 분리해 비교한다.

### CP34-M 이후 모델 역할 분리

CP34-M에서는 모델 구조를 새로 늘리지 않고 예측선 역할과 밴드 역할을 분리했다. 같은 모델이라도 line 후보와 band 후보를 같은 기준으로 탈락시키지 않고, 평가판을 분리해 판단한다.

| 모델 | 현재 역할 | CP34-M 판정 |
|---|---|---|
| PatchTST | line 후보로 보류 | clean feature 기준 direct band preset은 탈락/보류 |
| TiDE | band 후보 1번 | q25-b2 50티커 smoke에서 coverage 부족과 breach 과다로 탈락 |
| CNN-LSTM | band 후보 2번, 빠른 smoke/calibration용 | q25-b2 50티커 smoke에서 coverage 부족과 upper breach 과다로 탈락 |
| DLinear/NLinear | 다음 단순 baseline 후보 | 이번 CP에서는 구현하지 않음 |
| NHITS/N-BEATS | Phase 1.5 후보 | 이번 CP 범위 밖 |

CP34-M에서 구조 변경은 하지 않았다. TiDE의 future covariate 구조와 CNN-LSTM의 빠른 학습 특성은 여전히 검토 가치가 있지만, q25-b2 smoke 기준으로는 밴드 후보 생존 조건을 통과하지 못했다. 이후 다시 확인한다면 q20-b2처럼 더 넓은 preset 또는 lambda 조정만 최소 범위로 확인해야 하며, 대규모 head 추가나 target 변경은 금지한다.

### CP35-M 이후 TiDE/CNN-LSTM 역할 조정

CP35-M에서도 모델 구조는 변경하지 않았다. TiDE는 q10-b2 direct와 q10-b2 param을 확인했고, CNN-LSTM은 seq_len 120/60 짧은 lookback을 확인했다.

| 모델 | 확인한 구조적 가정 | 결과 | 현재 역할 |
|---|---|---|---|
| TiDE | 더 보수적인 q10/q15 밴드, param 양수 폭 밴드 | q10-b2 param은 validation band 기준만 약한 보류 | band 전용 gate 분리 후 재검토 가능 |
| CNN-LSTM | seq_len 120/60으로 local volatility band 확인 | coverage 0.72대까지 회복했지만 upper breach와 line 지표 실패 | 현재 band 후보 탈락, 빠른 smoke용 보조 |

중요한 구조적 관찰은 `coverage_gate`가 line 지표까지 포함한다는 점이다. CP34에서 평가판을 분리했으므로, band 모델 후보를 별도로 검증하려면 checkpoint selector도 band 전용으로 분리되어야 한다. 그렇지 않으면 TiDE q10-b2 param처럼 밴드 자체는 기준에 들어오는 후보도 line 실패 때문에 fallback으로 저장된다.

### CP36-M 이후 checkpoint selector 구조

CP36-M부터 모델 구조와 checkpoint selector 역할을 분리한다. 모델 architecture 자체는 변경하지 않았고, 어떤 epoch를 저장할지 고르는 정책만 분리했다.

| selector | 모델 역할 | 통과 기준 |
|---|---|---|
| `line_gate` | 예측선 후보 | IC/spread/MAE/SMAPE |
| `band_gate` | 밴드 후보 | coverage/breach/width/band loss |
| `combined_gate` | line + band 동시 후보 | line_gate와 band_gate 모두 통과 |
| `coverage_gate` | deprecated alias | 내부적으로 combined_gate |

이제 TiDE 같은 band 후보는 `role=band_model`로 저장될 수 있고, PatchTST 같은 line 후보는 `role=line_model`로 저장될 수 있다. `band_gate` 통과 여부는 line 지표 실패와 독립이며, `line_gate` 통과 여부도 band 과보수 문제와 독립이다.

구조적으로 중요한 점은 이 변경이 head, target, loss, 모델 forward를 바꾸지 않는다는 것이다. CP36은 모델 성능 개선이 아니라 실험판 분리다.

### CP37-M 이후 역할별 후보

CP37-M에서 같은 clean feature 기준으로 `line_gate`와 `band_gate`를 실제 실험에 적용했다. 그 결과 모델 역할은 다음처럼 정리한다.

| 역할 | 후보 | 상태 | 근거 |
|---|---|---|---|
| line_model | PatchTST | 생존 | `line_gate_pass=true`, IC/spread 양수 |
| band_model | TiDE param | 보류 | validation `band_gate_pass=true`, test coverage 낮음 |
| band_model | TiDE direct | 보류 | validation `band_gate_pass=true`, test coverage 낮음 |
| band_model | CNN-LSTM seq60 | 보류 | validation `band_gate_pass=true`, test upper breach 약간 초과 |
| band_model | CNN-LSTM seq120 | 탈락 | validation `band_gate_pass=false` |

PatchTST는 direct band 후보로는 여전히 약하지만 line 후보로는 살아 있다. TiDE/CNN-LSTM은 band 후보로 일부 살아났지만, test split에서 coverage가 낮아 구조 확장보다 split 안정성 확인이 먼저다.

### CP38-M 이후 band calibration 구조

CP38-M에서는 모델 구조를 바꾸지 않고 band output 후처리 보정층만 검토했다. 추가한 도구는 `ai/band_calibration.py`이며, checkpoint의 val/test 예측을 다시 모아 calibration을 적용한다.

검토한 calibration 방식:

| 방식 | 설명 | 모델 구조 영향 |
|---|---|---|
| scalar width calibration | line 기준 lower/upper 폭에 validation 기반 scale을 곱함 | 없음 |
| conformal residual calibration | validation residual 분위수로 line 기준 offset 산출 | 없음 |

결과적으로 CNN-LSTM seq60은 scalar width calibration에서 test coverage 0.829036, upper breach 0.078392, lower breach 0.092571로 band 후보 기준을 통과했다. TiDE param은 lower breach가 남아 단순 보정으로 해결되지 않았다.

현재 역할 구조:

- line_model: PatchTST
- band_model: CNN-LSTM seq60 + scalar width calibration
- TiDE param/direct: 보류에서 탈락 쪽으로 이동. 단, 다른 split 안정성 실험 전 완전 폐기는 보류한다.

### CP39-M 이후 역할 조합 구조

CP39-M에서 100티커 기준 역할 조합 가능성이 확인됐다. 모델 구조는 여전히 분리되어 있다.

| 역할 | 모델 | 입력 길이 | 출력 계약 | 상태 |
|---|---|---:|---|---|
| line_model | PatchTST q25-b2 | 252 | 1D horizon=5 raw return line | 생존 |
| band_model | CNN-LSTM seq60 q20-b2 direct + scalar width calibration | 60 | 1D horizon=5 raw return lower/upper band | 생존 |

두 모델의 seq_len은 다르지만 `feature_version=v3_adjusted_ohlc`, `timeframe=1D`, `horizon=5`, `raw_future_return` 출력 공간이 같다. 따라서 같은 ticker/asof_date/horizon에 대해 inference 단계 조합이 가능하다.

향후 저장 구조는 단일 `run_id`만으로 부족하다. 조합 예측에는 최소 `line_model_run_id`, `band_model_run_id`, `band_calibration_method`, `band_calibration_params`가 필요하다.

`conservative_series` 초안은 long-only 기준으로 calibrated lower band를 사용한다. short 문맥을 열 경우 `conservative_short_series=calibrated_upper_band_series`를 별도로 둔다.

### CP40-M 조합 prediction 구조

CP40-M부터 line/band 조합 output의 계약은 다음처럼 둔다.

| prediction 필드 | 소스 |
|---|---|
| `line_series` | PatchTST line model |
| `lower_band_series` | CNN-LSTM band model + scalar width calibration |
| `upper_band_series` | CNN-LSTM band model + scalar width calibration |
| `conservative_series` | calibrated lower band |
| `band_quantile_low/high` | band model checkpoint의 q_low/q_high |
| `meta.line_model_run_id` | line 모델 run 또는 checkpoint ref |
| `meta.band_model_run_id` | band 모델 run 또는 checkpoint ref |
| `meta.band_calibration_params` | scalar width calibration 계수 |
| `meta.prediction_composition_version` | `line_band_v1` |

두 모델의 `seq_len`은 달라도 되지만 `feature_version`, `timeframe`, `horizon`, `line_target_type`, `band_target_type`은 같아야 한다. `ticker`, `asof_date`, `forecast_dates`가 맞지 않으면 조합 inference는 실패해야 한다.

이번 CP에서는 `lower <= upper`만 hard contract로 두고, `lower <= line <= upper`는 기록만 한다. line과 band가 서로 다른 모델에서 오기 때문에 중심선이 밴드 밖에 나갈 수 있으며, 이를 강제로 보정할지는 다음 calibration/policy CP에서 결정한다.

### CP41-S 실제 저장 run 조합 구조

CP41-S에서 composite prediction도 하나의 `model_runs` row로 저장한다.

| 역할 | 저장 위치 |
|---|---|
| line 모델 추적 | `predictions.meta.line_model_run_id` |
| band 모델 추적 | `predictions.meta.band_model_run_id` |
| 조합 run 추적 | `predictions.run_id = composite run_id` |
| calibration 추적 | `predictions.meta.band_calibration_method`, `predictions.meta.band_calibration_params` |

Composite `model_runs.model_name`은 `line_band_composite`이고, `band_mode`는 `composite_direct`로 저장한다. checkpoint는 새로 만들지 않으므로 `checkpoint_path`는 비워 둔다.

실제 검증 run은 `composite-1D-a0786769a07a`이며, line source는 `patchtst-1D-41d584bcb3cb`, band source는 `cnn_lstm-1D-76f363b84218`이다. 이 구조로 `predictions`, `prediction_evaluations`, `backtest_results` 저장이 모두 통과했다.

### CP42-M 200티커 역할 구조 판정

200티커 제한 검증에서도 모델 역할 분리는 유지한다.

| 역할 | 모델 | 구조 | 200티커 판정 |
|---|---|---|---|
| line_model | PatchTST | seq_len 252, patch_len 16, stride 8, q25-b2 | 생존 |
| band_model | CNN-LSTM | seq_len 60, q20-b2 direct, scalar width calibration | 생존 |

CNN-LSTM 원본 밴드는 200티커에서 coverage 0.483245로 너무 좁아 실패했다. 따라서 band 모델은 반드시 scalar width calibration 후처리와 함께 사용해야 한다.

Composite probe에서 `line_inside_band_ratio`는 약 0.516이었다. 즉, line과 band를 다른 모델에서 가져오는 구조는 가능하지만 line이 항상 band 내부에 들어간다는 보장은 없다. 현재 hard contract는 `lower <= upper`이며, `line_inside_band`는 기록 지표로 유지한다.

다음 구조 결정 후보:

- band를 PatchTST line 중심으로 recenter한다.
- `conservative_series=min(line_series, lower_band_series)`로 더 보수화한다.
- 현재처럼 calibrated lower band를 conservative_series로 유지하고, line_inside_band는 진단 지표로만 둔다.

### CP43-M composite policy 구조 결정

CP43-M에서 line/band composite 후처리 정책을 비교했다. 입력 모델은 그대로 유지한다.

| 역할 | 모델 | checkpoint |
|---|---|---|
| line_model | PatchTST q25-b2 line_gate | `patchtst_1D_patchtst-1D-d521eff215b1.pt` |
| band_model | CNN-LSTM seq60 q20-b2 direct + scalar width calibration | `cnn_lstm_1D_cnn_lstm-1D-5a347fab1538.pt` |

비교 결과 `risk_first_lower_preserve`가 다음 composite 정책 후보로 올라갔다. 정책 수식은 raw return 공간에서 `lower=min(calibrated_lower,line)`, `upper=max(calibrated_upper,line)`로 둔다. 이 정책은 `lower <= line <= upper`를 보장하고, CP42 raw composite 대비 avg_band_width를 약 2.94%만 늘리면서 coverage 0.870213, upper breach 0.122340을 기록했다.

`line_centered_asymmetric`은 CNN-LSTM의 하방폭/상방폭을 PatchTST line 중심으로 옮겼지만 coverage 0.374468, upper breach 0.562766으로 실패했다. 따라서 단순 recentering은 현 구조에서 사용하지 않는다.

### CP44-D indicator-only ATR ratio 계약

CP44-D에서 `atr_ratio`는 차트 보조지표용 indicator로만 다룬다. 모델 구조, 입력 feature 수, target, loss, checkpoint selector는 변경하지 않았다.

계약:

| 항목 | 상태 |
|---|---|
| `FEATURE_COLUMNS` | 변경 없음 |
| `MODEL_N_FEATURES` | 36 유지 |
| `ai/preprocessing.py` feature list | 변경 없음 |
| `build_features()` output | indicator upsert를 위해 `atr_ratio` 포함 |
| 학습 입력 | `atr_ratio` 미포함 |

`atr_ratio`는 `ATR14 / close`이며 adjusted OHLC 계약 위에서 계산된다. 이 값은 `public.indicators`와 read-only indicator API에는 내려가지만, 모델 입력에는 들어가지 않는다. 향후 모델 피처로 승격하려면 별도 CP에서 feature contract version bump, cache 무효화, finite/p99/max contract, 1W/1M 극단값 clipping 또는 winsorization 정책을 먼저 정의해야 한다.

### CP44-M composite 저장 정책

Composite inference 저장 경로는 다음 정책을 지원한다.

| 정책 | 저장 경로 지원 | 설명 |
|---|---|---|
| `raw_composite` | 지원 | CP42 기준선. line이 밴드 밖에 있을 수 있음 |
| `include_line_clamp` | 지원 | line 포함을 위해 lower/upper를 확장 |
| `risk_first_lower_preserve` | 기본값 | 하방 보수성 우선. `lower=min(lower,line)`, `upper=max(upper,line)` |

`line_centered_asymmetric`은 CP43-M에서 탈락했으므로 저장 경로 기본 후보에서 제외했다.

저장 meta에는 `composition_policy`, `line_model_run_id`, `band_model_run_id`, `band_calibration_method`, `band_calibration_params`, `prediction_composition_version`을 기록한다. CP44-M smoke run `composite-1D-3a44b5e51ed2`에서 `lower <= line <= upper`와 meta 저장이 모두 통과했다.

### CP46-M composite upper buffer 정책

CP46-M에서 CNN-LSTM band 후보 `s60_q15_b2_direct_188`을 대상으로 composite 상단 보정을 비교했다. 기존 `risk_first_lower_preserve`는 line을 밴드 안에 넣지만 test upper breach가 0.285106으로 너무 컸다.

새 생존 정책:

| 정책 | 수식 | 역할 |
|---|---|---|
| `risk_first_upper_buffer_1.10` | `lower=base_lower`, `upper=line+(base_upper-line)*1.10` | 하방 보수성 유지, 상단만 10% 확장 |

이 정책은 `lower <= line <= upper`를 유지하고, conservative_series 기반 lower를 변경하지 않는다. test 기준 coverage 0.856383, upper breach 0.143617, width increase 1.087218로 통과했다.

따라서 현재 composite 후보 구조는 다음과 같다.

| 역할 | 후보 |
|---|---|
| line_model | PatchTST completed run |
| band_model | CNN-LSTM `s60_q15_b2_direct_188` |
| band calibration | scalar width |
| composition policy | `risk_first_upper_buffer_1.10` |

### CP48-M h20 branch architecture note

horizon=20은 h5 composite와 같은 후보군으로 섞지 않는다. CP48-M smoke에서 PatchTST line과 CNN-LSTM band 모두 h20 checkpoint 생성은 가능했지만, role gate 기준으로는 본류 후보가 아니었다.

| horizon | line model | band model | 상태 |
|---|---|---|---|
| h5 | PatchTST | CNN-LSTM scalar calibrated band + composite upper buffer | 본류 유지 |
| h20 | PatchTST smoke | CNN-LSTM smoke | Phase 1.5 보류 |

h20 composite는 `forecast_dates`와 예측 series 길이 20을 정상 생성했다. 다만 `composite_inference` 내부 contract check 이름이 `series_length_all_5`로 남아 있어 h20에서는 false가 찍힌다. 실제 길이 검증은 별도 확인했고, 향후 h20을 본류로 올릴 때 이 체크는 horizon 일반화가 필요하다.

h20 현재 실패 원인은 저장 계약이 아니라 모델 지표다. PatchTST h20 line은 IC와 spread가 음수였고, CNN-LSTM h20 band는 scalar calibration 후에도 composite upper breach가 과도했다.

### CP49-M PatchTST horizon rescue note

PatchTST 구조 자체는 변경하지 않았다. CP49-M은 모델 구조 변경이 아니라 horizon별 line 평가판 확장이다.

비교 축은 다음으로 고정했다.

| 축 | 값 |
|---|---|
| horizon | 5, 10, 20 |
| geometry | patch_len/stride 16/8, 32/16, 16/4 |
| seq_len | 252 기본, h20에서 504 1회 |
| selector | line_gate |

보수적 line 후보 판단을 위해 `false_safe_rate`, `downside_capture_rate`, `severe_downside_recall`, `conservative_bias`, `upside_sacrifice`를 추가했다. IC/spread가 약해도 false safe가 낮고 downside capture가 좋으면 risk 보조 line으로 보류할 수 있다.

CP49-M 결과 기준 역할 업데이트:

| 역할 | 후보 | 상태 |
|---|---|---|
| 제품 기본 line | PatchTST h5 longer_context, patch_len 32, stride 16 | 생존 |
| 기존 비교 기준 | PatchTST h5 baseline, patch_len 16, stride 8 | 보류 |
| 보수 risk-only line | PatchTST h5 dense overlap, patch_len 16, stride 4 | 보류 |
| h20 branch | PatchTST h20 longer_context, patch_len 32, stride 16 | Phase 1.5 보류 |
| h20 seq504 | PatchTST h20 baseline seq_len 504 | 탈락 |

### CP51-M 평가 계층 분리

모델 구조는 변경하지 않았다. 대신 평가 계층에서 line, band, composite 표시 지표를 분리했다.

| 계층 | 역할 | 대표 지표 |
|---|---|---|
| line | 하방 보수적 예측선 | `spearman_ic`, `long_short_spread`, `false_safe_rate`, `severe_downside_recall`, `conservative_bias` |
| band | calibrated risk interval | `nominal_coverage`, `empirical_coverage`, `coverage_error`, `interval_score`, `band_width_ic`, `downside_width_ic` |
| composite 표시 | 제품 출력 보조 검증 | `line_inside_band_ratio`, `line_inside_band_point_ratio`, `product_display_warning_rate`, `conservative_series_false_safe_rate` |

band horizon bucket은 line bucket과 충돌하지 않도록 `all_horizon_band_*`, `h1_h5_band_*`, `h6_h10_band_*`, `h11_h20_band_*` prefix를 사용한다. `line_inside_band_ratio`는 후보 탈락 기준이 아니라 화면 표시 안정성 지표다.

### CP52-M 메트릭 계약

모델 구조 변경 없이 평가 계약만 고정했다. line은 하방 보수적 예측선, band는 calibrated risk interval, composite는 제품 표시 정책 계층으로 분리한다.

| 계층 | hard gate 여부 | v1 판단 축 |
|---|---|---|
| line | hard gate 아님 | IC/Spread 안정성, false safe, severe recall, downside capture |
| band | hard gate 아님 | nominal 대비 coverage error, asymmetric interval score, dynamic width |
| composite | 모델 탈락 기준 아님 | 표시 정합성, conservative series false safe, width increase |

`false_safe_rate`는 deprecated alias로 보고, 명시 지표는 `false_safe_negative_rate`, `false_safe_tail_rate`, `false_safe_severe_rate`를 사용한다. train split 기반 threshold는 checkpoint config와 metrics에 남겨 inference/recheck에서 재사용할 수 있게 했다.

### CP53-M 재채점 이후 역할 구조

CP53-M 재채점 이후 line/band/composite 역할은 그대로 분리 유지한다. PatchTST는 line 후보, CNN-LSTM은 band 후보, composite는 제품 표시 정책 계층이다.

| 역할 | 현재 후보 | CP53 해석 |
|---|---|---|
| h5 line | PatchTST `h5_longer_context_seq252_p32_s16` | ic_mean과 spread가 양수이고 false-safe 균형이 가장 좋아 기본 line 후보 유지 |
| risk-only line | PatchTST `h5_dense_overlap_seq252_p16_s4` | IC/spread가 음수라 주 line 후보가 아니라 하방 위험 보조지표 후보 |
| band | CNN-LSTM `s60_q15_b2_direct` / `s60_q15_b2_direct_188` | nominal 대비 coverage_abs_error는 아직 크지만 dynamic width가 양수라 생존 |
| band 보류 | TiDE param/direct | coverage error와 downside_width_ic가 약해 composite 주력 후보에서 제외 |
| composite policy | `risk_first_upper_buffer_1.10` | 제품 표시 정합성은 좋지만 width 증가가 커서 모델 성능 지표가 아니라 정책 비용으로 기록 |

h10과 h20은 h5 제품 후보와 직접 경쟁시키지 않는다. h10은 별도 branch, h20은 Phase 1.5 branch로 유지한다.

### CP54-M baseline 비교 이후 역할 해석

CP54-M 이후에도 line/band/composite 역할 분리는 유지한다. 다만 band 후보의 해석은 낮춰 잡는다.

| 역할 | 후보/기준 | CP54 해석 |
|---|---|---|
| h5 line | PatchTST `h5_longer_context_seq252_p32_s16` | zero, momentum, reversal, historical mean 기준선을 대체로 이긴다. line model 후보 유지 |
| line baseline | shuffled momentum, reversal, historical mean | IC 일부는 양수지만 false-safe 보수성이 약하다. AI line의 최소 비교 기준 |
| band | CNN-LSTM `s60_q15_b2_direct_188` | dynamic width는 양수지만 rolling historical quantile/Bollinger보다 interval_score와 calibration이 약하다. 생존이 아니라 개선 필요 |
| band baseline | rolling historical quantile w252, Bollinger return w60 k1 | CP52 지표판에서 강한 기준선이다. 다음 band 모델은 이 기준선을 이겨야 한다 |
| composite policy | `risk_first_upper_buffer_1.10` | 제품 표시 정책으로만 유지한다. 모델 성능 우위 근거로 쓰지 않는다 |

따라서 현재 구조에서 PatchTST는 line 후보로 유지하지만, CNN-LSTM band는 “AI가 baseline을 이긴 band”가 아니다. band 계층은 baseline-aware calibration 또는 rolling quantile/Bollinger 대비 개선을 먼저 요구한다.

### CP55-M 이후 후보군 해석

CP55-M은 새 모델 구조를 추가하지 않고 기존 checkpoint/metrics 산출물을 후보별 동일 조건의 통계 baseline과 다시 비교했다. 실행은 경량화되어 `discover-only`, `smoke`, `max-candidates` 제한 모드로 분리됐다.

모델 역할 해석은 다음처럼 고정한다.

| 역할 | 현재 해석 | 근거 |
|---|---|---|
| line model | PatchTST h5 longer-context 유지 | `h5_longer_context_seq252_p32_s16`이 minimal line baseline 대비 IC와 severe recall에서 생존 |
| risk-only line | PatchTST dense overlap 보조 | false-safe는 매우 낮지만 IC/spread가 음수라 주 line 후보는 아님 |
| direct AI band | 보류 또는 탈락 | CNN-LSTM 후보는 CP52 전체 지표 부족으로 재실험 필요, TiDE/PatchTST band는 minimal 통계 baseline 대비 약함 |
| statistical band baseline | 강한 기준선 | rolling quantile/Bollinger return band가 interval_score와 coverage_abs_error에서 계속 강함 |

다음 band 구조는 “모델이 lower/upper를 직접 예측”하는 방향만 고집하지 않는다. rolling historical quantile 또는 Bollinger return band를 기본 위험 구간으로 두고, AI가 residual scale이나 regime별 보정값을 예측하는 baseline-aware band 구조를 우선 검토한다.

### CP56-M line/band/composite 아키텍처 감사

CP56-M 기준 아키텍처 역할은 계속 분리한다.

| 역할 | 현재 후보 | 감사 결론 |
|---|---|---|
| line model | PatchTST h5 longer-context | 유지. 단, line 전용 run의 band output은 제품 band로 해석하지 않는다. |
| band model | CNN-LSTM s60 q15-b2 direct, TiDE param/direct | 유지. 단, band center와 PatchTST line center 불일치를 계약상 명시해야 한다. |
| composite policy | risk-first 계열 | 제품 표시 정책이다. 모델 성능 지표와 분리한다. |

모델 출력 계약은 모두 `ForecastOutput(line, lower_band, upper_band, direction_logit optional)`로 맞아 있다. PatchTST는 RevIN denorm을 line/lower/upper에 적용하고, geometry는 checkpoint config로 복원된다. CNN-LSTM은 `fp32_modules=lstm,heads`와 cuDNN off 경로로 CUDA 안정화 계약이 있다. TiDE는 future covariate와 per-horizon decoder 경로를 checkpoint config로 복원한다.

현재 아키텍처 병목은 composite 중심선 계약이다. CNN-LSTM scalar width calibration은 CNN-LSTM 자체 line을 중심으로 lower/upper 폭을 조정한 뒤 PatchTST line과 합친다. 따라서 band 단독 평가는 통과해도 composite upper breach가 커질 수 있다. 다음 구조 변경 전에는 `width_only` 또는 line-centered calibration 계약을 먼저 명확히 해야 한다.

### CP57-M AI indicator layer 아키텍처

CP57-M 이후 Phase 1 아키텍처에서 composite model은 본류가 아니다. 역할은 아래처럼 분리한다.

| layer | 역할 | 제품 사용 output |
|---|---|---|
| AI line layer | 보수적 예측선 또는 line score | `line_series` |
| AI band layer | calibrated risk interval | `lower_band_series`, `upper_band_series` |
| overlay bundle | 여러 layer를 화면에서 동시에 표시 | layer provenance와 mismatch 표시 |

`line_model`의 lower/upper는 제품 band가 아니고, `band_model`의 line은 제품 예측선이 아니다. 부수 출력은 evaluation 내부 참고로만 둔다.

기존 `line_band_composite`와 composite policy 도구는 legacy/experimental 위치로 내린다. Phase 1 제품 아키텍처는 최신 completed line run과 최신 completed band run을 각각 조회하고, 화면에서 provenance와 함께 overlay하는 방향이다.

### CP59-P 제품 조회 계약

Phase 1 제품 기본 조회에서는 `line_band_composite`, `role=composite_model`, `deprecated_for_phase1_product_contract=true` run을 최신 completed 후보에서 제외한다. legacy composite는 삭제하지 않고 research/demo artifact로 보존하지만, 제품 기본 모델 또는 최신 정상 모델처럼 노출하지 않는다.

백엔드 `/api/v1/ai/runs`는 기본적으로 legacy composite를 제외하고, `include_legacy=true`가 명시된 경우에만 포함한다. 화면은 당분간 PatchTST line 중심 기본 후보를 사용하며, overlay bundle API가 생기기 전까지 line layer와 band layer를 하나의 composite 모델처럼 설명하지 않는다.

### CP60-M Composite Legacy Guard

Phase 1 모델 아키텍처 계약에서 `line_band_composite`는 제품 모델이 아니다. `ai/composite_inference.py`와 `ai/composite_policy_eval.py`는 legacy/research utility이며, PatchTST line과 CNN-LSTM/TiDE band를 하나의 모델 출력으로 합치지 않는다.

제품에서 사용하는 산출물은 `AI line layer`의 `line_series`와 `AI band layer`의 `lower_band_series`/`upper_band_series`다. `line_inside_band_ratio`는 표시 진단값이며 모델 탈락 기준이 아니다. `include_line_clamp`와 `risk_first_lower_preserve`는 Phase 1 본류 정책이 아니라 legacy 표시 진단 정책으로만 남긴다.

### CP61-M Metric Layer Architecture

평가 schema도 아키텍처 역할과 맞춰 분리했다. `line_metrics`는 line layer 전용이고 `band_metrics`는 band layer 전용이다. `legacy_overlay_diagnostics`는 overlay 표시/과거 composite 진단 전용이며 모델 랭킹에 사용하지 않는다.

같은 `ForecastOutput`이 line/lower/upper를 모두 내더라도 제품 역할은 config `role`과 selector로 결정한다. `line_gate`는 IC/spread/false-safe 계열만 보고, `band_gate`는 calibration/interval/dynamic width 계열만 본다.

### CP62-M Candidate Regrade Architecture

기존 후보 재채점도 CP61 layer schema를 따른다. Line 후보 표는 `line_metrics`만 사용하고, band 후보 표는 `band_metrics`만 사용한다. Composite/overlay 계열 값은 `legacy_overlay_diagnostics`로만 분류하며 모델 랭킹에는 들어가지 않는다.

PatchTST는 현재 line layer 후보로 남고, CNN-LSTM/TiDE/PatchTST band 후보는 통계 baseline과 별도로 비교된다. Band 후보가 line 지표를 못 내거나 line_inside_band가 낮아도 그 자체로 탈락시키지 않으며, 반대로 composite 표시 정책으로 band 후보를 살리지도 않는다.

### CP63-BM Band Feature Set Architecture

AI band layer의 입력 feature set은 line layer와 섞지 않는다. CP63-BM 기준 제품 band 후보는 `role=band_model`이며, 산출물은 `lower_band_series`와 `upper_band_series`다. band_model의 line 출력은 제품 예측선으로 쓰지 않고, feature pruning 판단에도 line IC/spread를 사용하지 않는다.

현 36개 `MODEL_FEATURE_COLUMNS`는 8개 그룹으로 관리한다.

| 그룹 | 대표 피처 | band layer 해석 |
|---|---|---|
| price/return | `log_return`, `open_ratio` | 직전 가격 충격과 갭 위험 |
| volatility/range | `high_ratio`, `low_ratio` | 상하단 range와 breach 위험 |
| volume/liquidity | `vol_change` | 정보 유입과 유동성 충격 |
| momentum/oscillator | `ma_*_ratio`, `rsi`, `macd_ratio`, `bb_position` | squeeze, 과열, trend regime |
| market/sector/breadth | macro, breadth, regime | 공통 volatility regime |
| fundamentals | revenue 계열 | 장기 체질 proxy지만 원시값 노이즈 큼 |
| calendar/future covariate | 요일, 월, 만기 | 알려진 시간 구조 |
| missingness flag | `has_*` | 가용성 proxy이며 편향 위험 있음 |

다음 band 실험의 기본 feature set은 `price_volatility`다. 이는 market/fundamentals를 제거하고 가격, range, momentum/oscillator만 남겨 CNN-LSTM s60 direct 계열이 통계 baseline 대비 약했던 원인이 feature noise인지 확인하기 위한 구조다.

`atr_ratio`와 `intraday_range_ratio`는 band 폭에 직접적인 indicator-only 후보지만 현재 feature contract에는 넣지 않는다. 이를 실제 모델 입력으로 추가하려면 `FEATURE_COLUMNS`, `MODEL_FEATURE_COLUMNS`, cache digest, feature version을 함께 변경해야 하며, 기존 checkpoint와 혼합하지 않는 별도 preset이 필요하다.

### CP64-BM Feature Set Smoke Architecture

CP64-BM에서는 AI band layer가 full 36개 feature만 쓰는 구조에서 벗어나, CP63 feature set plan의 부분집합을 CLI로 선택할 수 있게 했다. `ai.train`은 `--feature-set` 또는 `--feature-columns-preset`을 받아 `docs/cp63_bm_feature_set_plan.json`을 로드하고, 선택된 columns가 기존 `MODEL_FEATURE_COLUMNS`의 부분집합인지 검증한다. 모델 생성 시에는 `TrainConfig.n_features`를 사용해 CNN-LSTM/TiDE 입력 차원을 맞춘다.

feature set 적용은 데이터 bundle 생성 뒤 train/val/test feature tensor와 mean/std를 같은 column index로 축소하는 방식이다. 따라서 기존 full_features 기본 동작은 유지하면서도 `price_volatility`, `price_volatility_volume`, `no_fundamentals`, `technical_only` 같은 band 전용 부분집합을 같은 학습 경로에서 재사용할 수 있다.

이번 smoke 결과 BM 우선순위는 CNN-LSTM 1순위, TiDE 2순위로 유지한다. CNN-LSTM은 `price_volatility` q20과 `price_volatility_volume` q15가 band_survive로 남았고, TiDE는 `technical_only` param이 survive지만 interval score가 CNN-LSTM 상위 후보보다 약하다. PatchTST band는 계속 참고 후순위다.

`atr_ratio`와 `intraday_range_ratio`는 이번 CP에서 모델 feature로 추가하지 않았다. 해당 feature 승격은 feature contract, cache digest, feature_version, checkpoint 호환성 분리를 동반해야 하므로 CP64-BM architecture에는 포함하지 않고 별도 indicator/full backfill 또는 feature contract CP에서 다룬다.

### CP65-LM Line Feature Set Architecture

CP65-LM 기준 PatchTST line layer는 제품 `line_series` 산출물만 담당한다. 같은 `ForecastOutput`에 lower/upper가 있어도 이번 line_model 판정에는 쓰지 않고, band coverage나 lower/upper breach로 line 후보를 탈락시키지 않는다.

feature pruning은 기존 `MODEL_FEATURE_COLUMNS`의 부분집합만 사용한다. `technical_only`, `price_volatility_volume`, `no_fundamentals`는 CP63 feature set plan을 그대로 읽으며, `atr_ratio`와 `intraday_range_ratio`는 모델 feature로 추가하지 않았다. `technical_only`와 `price_volatility_volume`은 현재 동일한 11개 column 정의라 line architecture 관점에서는 같은 입력 공간으로 해석한다.

h5 product line은 기존 PatchTST `h5_longer_context_seq252_p32_s16` full_features를 주력 후보로 유지한다. h5 pruned 후보가 통계 baseline 기준 생존하더라도 full_features 대비 risk 지표를 개선하지 못하면 제품 line 후보를 교체하지 않는다.

h20은 h5와 같은 후보군으로 섞지 않고 Phase 1.5 visual/risk branch로 분리한다. 제품에서는 h5 line을 기본 예측선으로 두고, h20 line은 중기 위험 방향 보조선 또는 horizon 토글로 표시해야 한다. 두 horizon을 평균 내거나 같은 ranking 표에서 직접 경쟁시키지 않는다.

### CP64-D Indicator Architecture

CP64-D 기준 `atr_ratio`는 모델 feature가 아니라 indicator-only 출력이다. `feature_svc.build_features()`는 `atr_ratio`를 `public.indicators`에 저장할 수 있도록 output에 포함하지만, `FEATURE_COLUMNS`와 `MODEL_FEATURE_COLUMNS`에는 넣지 않는다. 따라서 `MODEL_N_FEATURES=36` 계약은 유지한다.

가격 파생 indicator는 adjusted OHLC 기준으로 재계산한다. raw `open/high/low/close`를 그대로 쓰지 않고 `adjusted_close / close` factor를 적용한 뒤 `open_ratio`, `high_ratio`, `low_ratio`, ATR, RSI, MACD, Bollinger 계열을 계산한다. 1D/1W/1M resample도 같은 adjusted OHLC 계약을 따른다.

`compute_indicators` full backfill은 이제 `force_full_backfill=True`에서 기존 ticker/timeframe/source 범위를 정리한 뒤 현재 `build_features()` 결과를 upsert한다. 이 경로는 stale indicator row 제거용 운영 backfill 계약이며, daily incremental cron에는 적용되지 않는다.

`atr_ratio`는 BM band feature 후보로는 준비됐지만 아직 architecture 상 모델 입력이 아니다. 승격하려면 indicator-only에서 feature column으로 옮기고, feature version/cache digest/checkpoint compatibility를 함께 분리해야 한다. `daily_range_ratio`와 `intraday_range_ratio`는 현재 DB column이 없으므로 먼저 이름과 계산식 계약을 정해야 한다.

### CP66-LM Post-backfill Line Architecture

CP66-LM 이후 line layer cache 계약은 source data hash와 feature contract를 분리해서 본다. CP64-D backfill 이후 1D source data hash는 `f7c7b101`로 바뀌었지만 feature contract는 `v3_adjusted_ohlc` 그대로다. 따라서 checkpoint 호환성 관점에서는 v4 feature contract를 만들지 않고, 같은 36개 모델 입력으로 post-backfill cache만 새로 물질화한다.

PatchTST h20은 h5 product line과 같은 후보군이 아니다. h20은 중기 line 또는 visual/risk branch이며, 제품에서는 h5 기본 line과 별도 horizon 보조선으로 표시해야 한다. CP66 기준 h20 `full_features`만 false_safe_tail_rate 0.2895로 0.30 미만이고 severe_downside_recall도 CP65 full 대비 개선되어 시각/risk branch 생존 후보가 됐다.

`no_fundamentals`와 `technical_only` h20은 IC/spread가 양수여도 false_safe_tail_rate가 0.30 이상이므로 제품 기본 표시는 금지한다. 이 경우 화면에 노출하더라도 기본 굵은 선이 아니라 낮은 신뢰도 점선, horizon 토글, 또는 연구용 중기 위험 보조선으로 제한한다.

1W line architecture는 readiness가 열렸지만 아직 제품 후보는 아니다. 1W는 `timeframe=1W`, `horizon=12`, `seq_len=104`가 통과 가능한 기본 형태이고 split gap은 h_max=12를 사용한다. 전체 eligible ticker registry가 default registry보다 넓어졌으므로 1W line을 제품 후보로 키우기 전에 registry 갱신 또는 hashed registry 사용 계약을 먼저 정리해야 한다.

### CP67-LM h20 제품 표시 Architecture

CP67-LM 100티커 재검증 이후 h20 line은 제품 기본 ON 후보가 아니다. 50티커에서는 CP66 `full_features`가 false_safe_tail_rate 0.2895, severe_downside_recall 0.7120으로 생존했지만, 100티커에서는 false_safe_tail_rate 0.3918, severe_downside_recall 0.6190으로 제품 보조 line 기준을 통과하지 못했다.

h20 line의 제품 표시 계약은 다음처럼 둔다.

| horizon | 표시 역할 | 기본 표시 |
|---|---|---|
| h5 | 단기 line / 제품 기본 예측선 | 진한 실선, 기본 ON |
| h20 | 중기 line / 위험 맥락 참고선 | CP67 기준 기본 OFF, Phase 1.5 연구 후보 |

h20은 IC/spread가 양수여도 false-safe가 높으면 제품 기본 표시를 금지한다. CP67 h11_h20 bucket은 IC 0.0633, spread 0.0203으로 ranking 자체는 무너지지 않았지만 false_safe_tail_rate 0.3829라 장기 구간 안전 오판 위험이 크다. 따라서 제품에서 h20을 보여주려면 별도 토글 또는 연구용 provenance가 필요하고, 기본 차트에는 h5 line만 둔다.

CP67은 모델 구조를 바꾸지 않았다. PatchTST 구조, feature contract, `MODEL_FEATURE_COLUMNS=36`, `v3_adjusted_ohlc`, `line_gate` 계약은 그대로 유지한다. `atr_ratio`는 여전히 indicator-only이며 line feature로 승격하지 않는다.

### CP68-LM h20 Conservative Line Architecture

CP68-LM 이후 h20은 `raw_line`과 `conservative_line`을 분리해서 해석한다. `raw_line`은 CP67 checkpoint의 기대 수익 line이며 IC 0.0667, spread 0.0280으로 cross-sectional ranking 신호는 있다. 그러나 false_safe_tail_rate 0.3918로 안전 오판이 높기 때문에 제품 위험선이나 기본 중기선으로 직접 쓰지 않는다.

`conservative_line`은 모델 구조를 바꾸지 않는 display calibration layer다. validation split에서 offset을 fit하고 test에는 고정 적용한다. CP68 best 정책은 `horizon_bucket_downshift`이며 h1_h5, h6_h10, h11_h20 bucket별 offset을 따로 둔다. 이 방식은 각 horizon bucket의 sign threshold를 보수적으로 낮추지만, bucket 내부 순위는 유지하므로 IC/spread 희생이 거의 없다.

제품 표시 계약은 다음처럼 갱신한다.

| horizon | 산출물 | 표시 역할 | 기본 표시 |
|---|---|---|---|
| h5 | `line_series` | 단기 line / 제품 기본 예측선 | 진한 실선, 기본 ON |
| h20 raw | `raw_line_series` 또는 연구 지표 | 랭킹 참고용 | 제품 위험선 사용 금지 |
| h20 conservative | `conservative_line_series` 후보 | 중기 보수 판단선 | 기본 OFF, 사용자 선택형 |

CP68은 post-hoc calibration 검증이며 PatchTST 구조, feature contract, `MODEL_FEATURE_COLUMNS=36`, `v3_adjusted_ohlc`, `line_gate`는 바꾸지 않았다. `atr_ratio`는 계속 모델 feature가 아니다. 실제 제품 연결은 아직 하지 않았고, UI/backend API도 수정하지 않았다.

### CP69-LM Line Terminology Contract

CP69-LM 이후 line architecture 문서에서는 “보수선”이라는 단어를 다음 세 용어로 나눠 쓴다.

| 용어 | 의미 | 생성 위치 |
|---|---|---|
| `raw_model_line` | checkpoint가 직접 출력한 line | 모델 forward |
| `trained_conservative_line` | alpha/beta asymmetric Huber loss로 학습된 line | 학습 loss |
| `display_calibrated_line` | validation offset으로 아래로 이동한 표시선 | post-hoc display calibration |

`AsymmetricHuberLoss`는 `prediction - target`이 양수인 과대예측에 `beta=2`를 적용한다. `ForecastCompositeLoss`는 이 손실을 `prediction.line`과 `line_target`에 직접 적용하므로, alpha=1 beta=2 config가 저장된 checkpoint의 원출력 line은 `trained_conservative_line`으로 분류한다.

CP69 감사 결과 h5 longer/baseline/dense와 CP65/CP67 h20 PatchTST checkpoint는 모두 `verified_trained_conservative`다. 반대로 CP68의 `global_downshift`, `horizon_bucket_downshift`, `volatility_scaled_downshift`는 모델 학습 손실이 아니라 validation offset 표시 보정이므로 `display_calibrated_line`이다. 제품 문서와 차트 계약에서 CP68 보정선을 “학습된 보수선”이라고 부르지 않는다.

### CP70-LM h20 Display Calibration Policy

CP70-LM 기준 h20 제품 표시 architecture는 `trained_conservative_line` 원출력과 `display_calibrated_line` 표시선을 분리한다. h20 raw model output은 alpha=1 beta=2 loss로 학습된 `trained_conservative_line`이지만, 100티커 test에서 false_safe_tail_rate가 0.3911로 높아 제품 표시선으로 직접 쓰지 않는다.

세 display calibration 후보 중 제품 후보는 `horizon_bucket_downshift`다. 이 정책은 h1_h5, h6_h10, h11_h20별 validation offset을 따로 적용한다. CP70 현재 cache 기준 test에서 IC 0.0676, spread 0.0284, false_safe_tail_rate 0.2841, severe_downside_recall 0.7186, h11_h20 false_safe_tail_rate 0.2630으로 기본 OFF 사용자 선택형 기준을 통과했다.

표시 정책은 다음처럼 둔다.

| line | 의미 | 제품 표시 |
|---|---|---|
| h5 `trained_conservative_line` | 단기 기본 line | 기본 ON, 진한 실선 |
| h20 `raw_model_line` | 중기 ranking 참고용 원출력 | 직접 표시 금지 |
| h20 `display_calibrated_line` | validation bucket offset 표시선 | 기본 OFF, 사용자 선택형 점선/반투명선 |

CP70은 CPU-only forward-only 재평가였고 GPU를 쓰지 않았다. 단 current source hash `3ac43945`가 CP67/CP68의 `f7c7b101`과 달라, 이 결과는 동일-hash 재현이 아니라 현재 cache에서의 정책 안정성 확인으로 본다.

### CP74-G W&B 초기화 계층

W&B 초기화는 모델 구조나 데이터 계층이 아니라 학습 관측성 계층으로 분리한다. full training의 모델 학습, checkpoint, save-run은 W&B online init 성공 여부에 의존하지 않는다. `ai.train`은 `WandbInitOutcome`을 통해 run 객체와 `wandb_status`를 분리하고, W&B가 실패하면 `wandb_run=None`으로 진행한다.

`wandb_status`는 train 결과 dict와 save-run config/meta JSON에 기록된다. `wandb_run_id`와 `wandb_run_url`은 `online_ok`일 때만 채운다. model_runs schema는 변경하지 않는다.

sweep은 관측성 누락을 조용히 허용하지 않는다. `ai.sweep`은 `--wandb`가 켜져 있는 trial을 `wandb_required=True`로 실행해 W&B init 실패를 trial 학습 전에 명시적인 오류로 다룬다. 따라서 full training fallback과 sweep required 정책은 같은 helper를 쓰되 required flag로 분리된다.

### CP75-G Local Training Logging Layer

local training progress logger는 모델 구조가 아니라 학습 관측성 계층이다. `ai.local_logging.LocalTrainingProgressLogger`가 `logs/runs/{run_id}/` 아래에 `config.json`, `metrics.jsonl`, `summary.json`을 기록한다.

`ai.train`은 epoch summary를 stdout과 W&B에 유지하면서 같은 핵심 지표를 local JSONL에도 남긴다. 파일 쓰기 실패는 학습 실패로 전파하지 않고 warning 후 local logger만 비활성화한다. 따라서 W&B optional fallback, checkpoint selection, loss, model forward 계약은 그대로 유지된다.

이 계층은 장기 full run의 관측성을 위한 보조 수단이며, DB 저장 계약이나 frontend 표시 계약을 대체하지 않는다.

### CP72-BM Band Gate와 Feature Set Inference

CP72-BM 이후 band layer의 checkpoint gate는 fixed coverage target이 아니라 `band_metrics.nominal_coverage`를 기준으로 해석한다. q15/q85 band의 nominal coverage는 0.70이며, empirical coverage가 이 값과 얼마나 가까운지와 lower/upper tail breach error가 과도하지 않은지를 본다. 이는 band가 보수적 line이 아니라 calibrated risk interval이라는 Phase 1 계약에 맞춘 것이다.

`band_gate`는 line 지표인 `spearman_ic`, `long_short_spread`, `line_inside_band`를 사용하지 않는다. `coverage_gate`와 `combined_gate`는 legacy 경로로 남아 있지만 CP72-BM 제품 후보 선택에는 사용하지 않는다.

`ai.inference`는 checkpoint config의 `feature_columns`와 `n_features`를 모델 복원과 입력 tensor 양쪽에 적용한다. `price_volatility_volume` checkpoint는 11개 feature channel을 기대하므로, DataLoader 생성 전에 full 36개 feature bundle에서 해당 column만 선택한다. 이 순서가 틀리면 CNN-LSTM conv 입력 channel mismatch가 발생한다.

CP72 최종 제품 후보 `cnn_lstm-1D-d0c780dee5e8`은 `role=band_model`, `feature_set=price_volatility_volume`, `n_features=11`, `band_mode=direct`, `q_low=0.15`, `q_high=0.85`, `lambda_band=2.0`인 CNN-LSTM band checkpoint다. 제품에서 쓰는 산출물은 `lower_band_series`와 `upper_band_series`이며, band_model의 line 출력은 제품 예측선으로 보지 않는다.
### CP75-LM h5 line 저장 후보 Architecture

CP75-LM의 제품 기본 line 후보는 PatchTST 1D h5 `line_model` 단독 run이다. 제품에서 사용하는 산출물은 `line_series`이며, line_model이 함께 출력하는 lower/upper 계열 값은 제품 band 후보로 쓰지 않는다. CP72 BM band 후보와 결합 저장하지 않았고, band/composite/overlay 계층도 건드리지 않았다.

해당 run `patchtst-1D-efad3c29d803`의 구조 계약은 `timeframe=1D`, `horizon=5`, `seq_len=252`, `patch_len=32`, `patch_stride=16`, `feature_set=full_features`, `n_features=36`, `feature_version=v3_adjusted_ohlc`, `checkpoint_selection=line_gate`, `target=raw_future_return`, `line_target_type=raw_future_return`이다. `atr_ratio`와 `intraday_range_ratio`는 indicator/cache에는 존재할 수 있어도 모델 feature로 승격하지 않았다.

loss 해석은 CP69 계약을 따른다. 이 후보는 alpha=1, beta=2, delta=1의 asymmetric Huber line loss로 학습된 `trained_conservative_line`이다. 다만 CP75 결과 기준으로 순위/수익 예측선 신호는 양수지만 false-safe 계열이 높으므로, 현재 architecture 문서에서는 제품 기본 h5 line 저장 후보이되 위험 회피선 품질은 `watch`로 분류한다.

저장 계약은 `model_runs.status=completed`, `config.role=line_model`, checkpoint 존재, `predictions.line_series` 저장, `prediction_evaluations` 저장으로 닫았다. `predictions` 전체 count는 DB statement timeout 때문에 inference stdout count와 샘플 `line_series` 조회로 검증했다.

## 2026-05-03 Phase 1 제품 표시 구조

Phase 1 제품 구조는 composite model이 아니라 AI indicator layer 구조다. line model과 band model은 서로 다른 보조지표로 취급한다. RSI와 MACD가 서로 다른 해석을 갖는 것처럼, 보수적 예측선과 AI 밴드도 하나의 중심선으로 억지 결합하지 않는다.

### 제품 layer 계약

| layer | 제품 이름 | 현재 run | 역할 |
|---|---|---|---|
| line layer | 1D 보수적 예측선 v1 | `patchtst-1D-efad3c29d803` | 향후 수익 방향과 보수적 판단선을 참고하는 line 지표 |
| band layer | 1D AI 밴드 v1 | `cnn_lstm-1D-d0c780dee5e8` | 예상 변동 범위와 위험 구간을 참고하는 band 지표 |
| 1W line | 1W 보수적 예측선 | 없음 | 준비 중 |
| 1W band | 1W AI 밴드 | 없음 | 준비 중 |

`line_band_composite`는 Phase 1 제품 기본 구조에서 제외한다. 과거 composite 저장/평가 유틸리티는 legacy 연구용으로만 남긴다. `line_inside_band` 같은 composite 진단 지표는 제품 품질 기준이 아니다.

### 차트 표시 구조

주식 보기 화면은 다음 layer로 구성된다.

| 화면 layer | 데이터 |
|---|---|
| 가격 차트 | OHLC 또는 line price |
| 보수적 예측선 | line model의 `conservative_series` 우선, 없으면 `line_series` |
| AI 밴드 | band model의 `lower_band_series`, `upper_band_series` |
| rolling history | 과거 저장 prediction history의 line/band series |
| 거래량 | 가격 차트 하단 bar, 토글 가능 |
| 하단 보조지표 | RSI, MACD, ATR, AI band width 등 |

CP79 이후 모든 chart input은 렌더 직전에 time/date 오름차순 정렬, 중복 제거, invalid 제거를 통과한다. 이 계약은 1W/1M 전환 때 lightweight-charts의 `data must be asc ordered by time` 오류를 막기 위한 필수 방어선이다.

예측 시작 marker는 prediction 기준일이 아니라 오늘 날짜 기준으로 표시한다. 모델 기준일은 우측 정보 패널에서 별도로 보여준다. 오늘 marker는 현재 viewport에서 오늘 날짜가 벗어나면 함께 사라진다.

### AI 모델 화면 구조

기존 `모델 학습` 화면은 run 콘솔처럼 보였기 때문에 `AI 모델` 화면으로 바꿨다. 기본 화면은 제품 모델 현황 4슬롯만 보여준다. 제품 슬롯을 클릭하면 같은 상세 패널에서 모델 구조, 사용 데이터, 목표 대비 실제 품질, 아쉬운 점을 설명한다.

이전 실험은 기본 노출하지 않는다. 사용자가 `예측선 실험 보기` 또는 `밴드 실험 보기`를 눌렀을 때만 실험 목록을 보여준다. 실험 row를 클릭하면 제품 모델과 같은 상세 패널 위치에서 해당 실험의 설명을 보여준다. NaN 실패, 계산 불가, composite 실험은 기본 제품 UI에서 숨긴다.

### 백테스트 화면 구조

백테스트는 Phase 1에서 포트폴리오 리밸런싱 실험이 아니라 단일 티커 룰 기반 전략 실험이다. 현재 Risk Guard v1과 Trend Guard v1은 UI/흐름 확인용 데모 전략이다. 실제 투자 전략 후보로 확정하지 않는다.

현재 전략 구조는 가격, RSI, 60일 추세, 보수적 예측선, AI 밴드 하단을 읽어 단일 티커에 대해 보유/현금 상태를 바꾸는 방식이다. 여러 종목 포트폴리오, 리밸런싱, 최적화는 Phase 2 backlog다.

### 남은 구조 과제

- Bollinger overlay를 제품 차트에 넣으려면 API에 `bb_upper`, `bb_middle`, `bb_lower` 가격 series가 필요하다. 현재 `bb_position`만으로는 가격 위 overlay를 만들지 않는다.
- 1W line/band 모델은 readiness와 데이터 계약은 봤지만 제품 후보가 없다.
- AI 모델 상세 설명은 아직 metric 표 중심이다. 목표/실제/차이 외에 실패 이유와 개선 방향을 더 자연어 리포트처럼 생성할 필요가 있다.
- 보조지표 패널은 기능적으로는 동작하지만, 밀도와 위계는 추가 UX 개선이 필요하다.

### CP86-D market data provider 구조

CP86-D 이후 가격 수집은 provider abstraction을 통한다. 현재 provider는 `yfinance`와 `eodhd`이며, 로컬 primary는 `MARKET_DATA_PROVIDER=yfinance`, fallback은 `MARKET_DATA_FALLBACK_PROVIDER=eodhd`로 지정한다. EODHD provider는 삭제하지 않고 검증/장애 대응용으로 유지한다.

yfinance provider는 `auto_adjust=False`를 고정해 raw `Open/High/Low/Close/Volume`과 `Adj Close`를 함께 가져온다. 모델 feature contract는 여전히 `v3_adjusted_ohlc`이며, `adjusted_factor = adjusted_close / close`로 adjusted OHLC를 재구성한 뒤 `log_return`, OHLC ratio, MA, RSI, MACD, Bollinger, ATR ratio를 계산한다. `MODEL_FEATURE_COLUMNS`와 feature 수 36은 변경하지 않았다.

가격 저장 전에는 adjusted factor finite/양수, adjusted high/low 정합성, 날짜 중복, required price finite, ratio p99/max sanity를 통과해야 한다. `price_data`에는 provider/source 컬럼이 없으므로 row-level provenance는 현재 schema로는 남기지 못하고, `sync_state` meta와 CP86 metrics JSON에 provider/fallback_used/quality gate 결과를 남긴다.
### CP96-D source-aware feature/cache guard

CP96-D 이후 feature cache fingerprint는 가격 row provenance뿐 아니라 indicator 값 checksum도 포함한다. 같은 ticker/date/count라도 indicator numeric value가 바뀌면 `source_data_hash`가 달라지고 기존 `features_*.pt`와 `feature_index_*.pt`가 재사용되지 않는다. 이 계약은 모델 구조 변경이 아니라 데이터 입력 신뢰성 계약이다.

모델 입력 feature 계약은 계속 `v3_adjusted_ohlc`, `MODEL_N_FEATURES=36`이다. `atr_ratio`는 indicator checksum과 차트 보조지표 검증에는 포함될 수 있지만, 아직 `MODEL_FEATURE_COLUMNS`에는 포함하지 않는다.

provider/source 계약은 다음과 같다.

| provider | price/indicator read source |
|---|---|
| yfinance | `source='yfinance'` |
| eodhd | `source='eodhd'` 또는 legacy `source IS NULL` |

backtest anchor close와 product price/indicator API도 같은 provider source를 본다. 따라서 yfinance/EODHD 병렬 row가 같은 ticker/date에 있어도 모델 평가와 화면 가격이 섞이지 않아야 한다.

1W/1M resample은 최신 일봉 기준으로 완성된 period만 사용한다. 현재 진행 중인 주/월 bucket은 feature build와 API display 양쪽에서 제외한다.

### CP116-S torch-first import 계약

Windows 제품/학습 환경에서는 torch를 쓰는 AI entrypoint가 `pandas`/`numpy`보다 먼저 `ai.torch_bootstrap.bootstrap_torch()`를 호출한다. 이 계약은 모델 구조 변경이 아니라 native CUDA DLL 로드 안정성 계약이다.

하위 모델 모듈(`ai.models`, loss, postprocess 등)은 torch library module로 유지하되, 제품 스크립트와 학습 스크립트는 bootstrap entrypoint를 통과해야 한다. data-only CP runner는 torch를 직접 또는 간접 import하지 않아야 하며, 데이터 확인과 모델 inference가 섞이는 리허설은 장기적으로 subprocess 분리 대상이다.

### CP124-BM band loss guard 구조

1W band layer의 손실 구조는 여전히 `ForecastCompositeLoss`를 사용한다. 기본 구성은 line Huber loss, lower/upper quantile pinball band loss, direct 모드에서만 쓰는 band cross penalty, 선택적 direction 보조 손실이다. 제품 band 평가는 line metric이나 composite metric이 아니라 band_metrics만 사용한다.

CP124에서 lower/upper quantile pinball loss에 선택적 가중치를 추가했다. `lower_band_loss_weight=1.0`, `upper_band_loss_weight=1.0`이 기본이라 기존 checkpoint와 학습 기본 동작은 유지된다. 하방 guard 실험은 lower quantile loss만 1.5 또는 2.0으로 키워 lower breach를 줄일 수 있는지 확인하는 방식이며, target 구조와 feature contract는 바꾸지 않았다.

width alignment는 아직 모델 구조에 연결하지 않았다. 기존 `lambda_width`는 호환용 설정값이고 실제 손실 계산에는 포함되지 않는다. dynamic width alignment나 downside width alignment를 제품 후보 검증에 쓰려면 별도 CP에서 명시적인 width target 또는 differentiable alignment loss를 설계해야 한다.
### Phase 1 모델 슬롯 구조

Phase 1 모델 아키텍처는 네 슬롯으로 단순화한다.

| 슬롯 | 역할 | Phase 1 기준 |
|---|---|---|
| 1D line | 단기 보수적 예측선 | 500티커 기준 v1 필요 |
| 1D band | 단기 AI 위험 범위 | 500티커 기준 v1 필요 |
| 1W line | 주간 보수적 예측선 | 500티커 기준 v1 필요 |
| 1W band | 주간 AI 위험 범위 | 500티커 기준 v1 필요 |

line layer와 band layer는 계속 분리한다. line model의 lower/upper 출력은 제품 band로 해석하지 않고, band model의 line 출력은 제품 예측선으로 해석하지 않는다. composite는 Phase 1 제품 기본 구조에서 제외한다.

500티커 기준을 통과하지 못한 기존 100티커 모델은 데모/초기 후보로만 남긴다. 개별 실험의 모델군, feature_set, metric, 실패 분류는 각 실험 계획서에서 관리한다. 프로젝트 전체 기준은 `docs/phase1_project_status.md`를 따른다.
