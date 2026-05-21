# CP3.5 지시서 — 모델 아키텍처 보강 + 출력 정합성 fix

## 배경

CP3까지 데이터 파이프라인·sufficiency·split이 닫혔다. CP4 (ticker embedding) 진입 전, 리뷰어 라운드에서 도출된 **모델 구조·출력 정합성 갭 5건**을 한 묶음으로 정리한다.

이 CP는 **본 학습 진입 전 마지막 구조 보강 단계**다. 이후 CP4부터는 모델 코드를 더 이상 갈아엎지 않고 ticker embedding·loss weight sweep 등 위에 얹는 작업만 수행한다.

원칙: **비용보다 fidelity 우선** (사용자 합의 D22). 시간이 더 들어도 논문 표준에 맞게 제대로 구현한다.

근거 문서:
- `docs/model_architecture.md` — 옵션·장단점·결정 (A-3/B-2/C-2/D-1/E-1/E-2 확정).
- `docs/training_hyperparameters.md` — 학습 파라미터 SoT.
- `docs/project_journal.md` — 결정 D21·D22 기록.

---

## 목표

1. **출력 정합성 fix (P1·P2)**:
   - P1-1: 추론 정렬에서 line head 제외.
   - P1-2: val·test·inference 단일 후처리 함수.
   - P2-3: 밴드 폭 손실 음수 방지.
2. **PatchTST 풀 구현 (B-2)**: RevIN + Channel Independence.
3. **CNN-LSTM 안정화 (C-2)**: LayerNorm + residual + attention pooling.
4. **TiDE 재구현 (D-1)**: ResidualBlock 기반 enc/dec + temporal decoder + lookback skip.
5. **출력 헤드 두 방식 구현 (A-3)**: 직접 q10/q90 출력 + center/log_half_width 파라미터화. 모델 init 인자로 선택 가능.
6. **Init 통일 (E-1)**: BERT 식 trunc_normal(std=0.02) 전 모델.
7. **Dropout 위치 표준화 (E-2)**: 모델별 표준 위치 보강.
8. **검수 양식 보강**: 보고서에 "핵심 구성요소 존재 여부 체크리스트" 필수 섹션.

---

## 예상 시간

| 단계 | 시간 |
|---|---|
| 출력 헤드 리팩토링 (A-3) + postprocess 함수 (P1-1, P1-2) + width loss fix (P2-3) | 1.5~2시간 |
| PatchTST 풀 구현 (B-2: RevIN + Channel Independence) | 2.5~3.5시간 |
| CNN-LSTM 안정화 (C-2) | 1~1.5시간 |
| TiDE 재구현 (D-1) | 2.5~3.5시간 |
| Init 통일 (E-1) + Dropout 위치 표준화 (E-2) | 0.5~1시간 |
| 테스트 (forward 형상, postprocess 정합성, fidelity 체크리스트) | 2~3시간 |
| Dry-run 검증 (split → forward → loss 한 번 통과) | 0.5~1시간 |
| 회귀 (기존 25 테스트 green 확인) | 0.5시간 |
| 보고서 작성 | 0.5시간 |
| **총 체감** | **11~16시간** |

기준: CPU 작업 위주, 풀 학습 미실행. 5060 Ti 16GB 환경에서 dry-run smoke만 수행.

---

## 권한 번들 (사전 승인)

**파일 쓰기 범위**:
- `ai/models/patchtst.py`, `ai/models/cnn_lstm.py`, `ai/models/tide.py`, `ai/models/common.py`
- `ai/models/revin.py` (신규)
- `ai/models/blocks.py` (신규 — ResidualBlock, AttentionPooling 공용)
- `ai/postprocess.py` (신규 — `apply_band_postprocess`)
- `ai/loss.py` (P2-3 fix + 출력 파라미터화 시 cross loss 분기)
- `ai/train.py`, `ai/inference.py` (postprocess 호출 통일)
- `tests/**`
- `docs/cp3.5_report.md` (종료 리포트, 신규)

**허용 명령**:
- `uv run pytest tests/`
- `uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5`
- `uv run python -m ai.train --dry-run --timeframe 1W --seq-len 104 --horizon 4`
- `uv run python -c "..."` (모델 forward 형상 점검 임시 스크립트)

**금지**:
- 학습 실제 실행 금지 (1 epoch 이상)
- DB 쓰기 금지 (predictions·model_run·indicators 등)
- `git push` 금지
- 기존 25 테스트 빨간색 만드는 회귀 금지 (동시에 통과해야 함)

**휴먼 개입 트리거**:
- 어느 모델 풀 구현이 예상 시간 2배 초과 (PatchTST 7시간+, TiDE 7시간+)
- Forward shape 불일치로 dataset → model 연결 실패
- 기존 테스트가 깨졌는데 원인 불명 30분 초과
- 메모리 폭발 (Channel Independence 적용 시 OOM)

---

## 실행 상세

### 1단계 — 출력 헤드 리팩토링 (A-3) + 후처리 통일 (P1-1, P1-2) + Width loss fix (P2-3)

#### 1.1 `ai/models/common.py` 변경

`MultiHeadForecastModel`에 출력 모드 인자 추가:

```python
class MultiHeadForecastModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        band_mode: str = "direct",  # "direct" or "param"
    ):
        super().__init__()
        self.horizon = horizon
        self.band_mode = band_mode
        self.line_head = nn.Linear(hidden_dim, horizon)
        if band_mode == "direct":
            # 직접 q10, q90 출력
            self.band_head = nn.Linear(hidden_dim, horizon * 2)
        elif band_mode == "param":
            # center, log_half_width 출력 → q10 = center - exp(log_half_width)
            self.band_head = nn.Linear(hidden_dim, horizon * 2)
        else:
            raise ValueError(f"unknown band_mode: {band_mode}")

    def build_output(self, hidden):
        line = self.line_head(hidden)
        band_raw = self.band_head(hidden).view(hidden.size(0), self.horizon, 2)
        if self.band_mode == "direct":
            lower = band_raw[..., 0]
            upper = band_raw[..., 1]
        else:  # "param"
            center = band_raw[..., 0]
            log_half_width = band_raw[..., 1]
            half_width = torch.exp(log_half_width)  # 항상 양수
            lower = center - half_width
            upper = center + half_width
        return ForecastOutput(line=line, lower_band=lower, upper_band=upper)
```

세 모델 (PatchTST/CNN-LSTM/TiDE)의 `__init__`에 `band_mode` 인자 전파.

#### 1.2 `ai/postprocess.py` 신규

```python
"""
Lens 모델 출력 후처리 — 학습·검증·추론 통일.

원칙:
- line head 출력은 그대로 보존 (β=2 비대칭 학습 신호 유지).
- band는 crossing 방지를 위해 sort (band_mode="direct"일 때만 필요).
- band_mode="param"인 경우 구조적으로 lower ≤ upper이므로 sort 무해 (재정렬해도 동일).

사용 지점: train.py validation/test loop, inference.py write path.
학습 loss는 raw 출력에 직접 걸어 grad가 sort를 통과하지 않게 한다.
"""
from __future__ import annotations
import torch

def apply_band_postprocess(
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """line은 그대로, band만 정렬해 lower ≤ upper 보장."""
    band = torch.stack([lower, upper], dim=-1)
    band_sorted, _ = torch.sort(band, dim=-1)
    return line, band_sorted[..., 0], band_sorted[..., 1]
```

#### 1.3 `ai/inference.py` 수정 (P1-1)

기존:
```python
# 잘못된 코드
sorted_vals, _ = torch.sort(torch.stack([line, q10, q50, q90]))
```

변경:
```python
from ai.postprocess import apply_band_postprocess
line_out, lower_out, upper_out = apply_band_postprocess(line, lower_band, upper_band)
# DB write에 line_out, lower_out, upper_out 사용. line_out은 모델 출력 그대로.
```

#### 1.4 `ai/train.py` 수정 (P1-2)

validation·test 루프에서 metric 계산 직전에 동일 함수 호출:

```python
from ai.postprocess import apply_band_postprocess
line_pp, lower_pp, upper_pp = apply_band_postprocess(line, lower_band, upper_band)
coverage = ((y >= lower_pp) & (y <= upper_pp)).float().mean()
avg_band_width = (upper_pp - lower_pp).mean()
```

**학습 loss는 raw 출력에 그대로 적용** (grad 보존). val·test·inference만 후처리 통일.

#### 1.5 `ai/loss.py` 수정 (P2-3)

`WidthPenaltyLoss`:
```python
class WidthPenaltyLoss(nn.Module):
    def forward(self, lower, upper):
        # 음수 폭 보상 방지: relu로 양수만 페널티
        return F.relu(upper - lower).mean()
```

`band_mode="param"` 모델에서는 lower ≤ upper 구조적 보장이라 cross penalty 손실 항이 사실상 0. `ForecastCompositeLoss`에서 `band_mode` 인자 받아 cross loss 분기:

```python
class ForecastCompositeLoss(nn.Module):
    def __init__(self, ..., band_mode: str = "direct"):
        self.band_mode = band_mode
        self.use_cross_loss = (band_mode == "direct")
        ...
    def forward(self, ...):
        total = self.lambda_line * line_loss + self.lambda_band * band_loss + self.lambda_width * width_loss
        if self.use_cross_loss:
            total = total + self.lambda_cross * cross_loss
        return total
```

---

### 2단계 — PatchTST 풀 구현 (B-2)

#### 2.1 `ai/models/revin.py` 신규

```python
"""
Reversible Instance Normalization (Kim et al., 2022).

학습·추론 시 forward 시작에서 per-instance per-channel normalize,
모델 출력 후 denormalize. 비정상 시계열에 강건.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))
        # per-forward stats 저장용 (denormalize에서 사용)
        self._mean = None
        self._std = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(mode)

    def _normalize(self, x):  # x: [B, L, C]
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def _denormalize(self, y):  # y: [B, H, ...] (horizon predictions)
        if self.affine:
            y = (y - self.beta[..., None]) / (self.gamma[..., None] + self.eps)
        # mean·std는 [B, 1, C] 형태 → horizon에 broadcast
        # y 출력이 단일 채널 (line, lower, upper 각각)이라면 channel 평균 std 사용
        # → 구체 dim 매칭은 구현에서 세밀히 처리
        ...
```

**주의**: PatchTST의 RevIN은 입력 [B, L, C] 정규화 → 모델 통과 → 출력 시 채널 평균 후 한 번의 denormalize. denormalize 차원 매칭은 구현 시 신중히 (forward shape 트레이스 후 결정).

#### 2.2 `ai/models/patchtst.py` 재구현

```python
class PatchTST(MultiHeadForecastModel):
    def __init__(
        self,
        n_features: int = 29,
        seq_len: int = 252,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        horizon: int = 5,
        dropout: float = 0.1,
        band_mode: str = "direct",
        use_revin: bool = True,
        channel_independent: bool = True,
    ):
        super().__init__(hidden_dim=d_model * self._num_patches(seq_len, patch_len, stride), horizon=horizon, band_mode=band_mode)
        self.n_features = n_features
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin
        self.channel_independent = channel_independent

        n_patches = self._num_patches(seq_len, patch_len, stride)
        self.revin = RevIN(n_features) if use_revin else None
        self.patch_proj = nn.Linear(patch_len, d_model)  # CI: 채널당 독립 projection (한 시계열의 patch_len → d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        # head는 super().__init__()이 만들어줌 (hidden_dim = d_model * n_patches)

        # init 통일 (E-1)
        self.apply(_init_weights)

    @staticmethod
    def _num_patches(seq_len, patch_len, stride):
        return ((seq_len - patch_len) // stride) + 1

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        # x: [B, L, C]
        B, L, C = x.shape
        if self.use_revin:
            x = self.revin(x, mode="norm")
        if self.channel_independent:
            # [B, L, C] → [B*C, L, 1]
            x = x.permute(0, 2, 1).reshape(B * C, L, 1)
            # patching: [B*C, n_patches, patch_len]
            patches = x.squeeze(-1).unfold(dimension=1, size=self.patch_len, step=self.stride)
            z = self.patch_proj(patches)            # [B*C, N, d_model]
        else:
            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches = patches.contiguous().view(B, patches.size(1), -1)
            z = self.patch_proj(patches)            # [B, N, d_model] — 단, patch_proj 차원 다름
            # (사용 안 함: channel_independent=True가 표준)
        z = z + self.position_embedding[:, : z.size(1)]
        z = self.input_dropout(z)
        z = self.encoder(z)                          # [B*C, N, d_model]
        z = self.norm(z)
        z = self.output_dropout(z)
        # flatten 후 head 입력: [B*C, N*d_model]
        z = z.reshape(z.size(0), -1)
        if self.channel_independent:
            # [B*C, N*d_model] → head → [B*C, horizon] (line) and [B*C, horizon, 2] (band raw)
            # 채널별 출력 → 채널 평균 (또는 attention pool)
            line_per_ch = self.line_head(z).view(B, C, self.horizon).mean(dim=1)
            band_per_ch = self.band_head(z).view(B, C, self.horizon, 2).mean(dim=1)
            # build_output을 직접 흉내
            if self.band_mode == "direct":
                lower, upper = band_per_ch[..., 0], band_per_ch[..., 1]
            else:
                center, lhw = band_per_ch[..., 0], band_per_ch[..., 1]
                hw = torch.exp(lhw)
                lower, upper = center - hw, center + hw
            line = line_per_ch
            if self.use_revin:
                # 채널 평균 출력에 대한 denormalize 필요
                # n_features 평균을 가진 mean·std로 처리 (구현 시 차원 careful)
                ...
            return ForecastOutput(line=line, lower_band=lower, upper_band=upper)
        else:
            return self.build_output(z)
```

**구현 메모**:
- `n_features=29`, `seq_len=252`, `patch_len=16`, `stride=8` → patches = 30개. `d_model=128`이면 head 입력 = 30 × 128 = 3840.
- Channel Independence + RevIN 동시 적용 시 메모리 ~ B × 29배. batch size 조정 필요할 수 있음 (smoke test에서 확인).
- denormalize 차원 매칭은 구현 시 신중히. 가능하면 `revin.mean.mean(dim=-1)`로 채널 평균 mean·std를 만들어 horizon에 broadcast.

#### 2.3 PatchTST forward shape 단위 테스트

```python
def test_patchtst_forward_shape():
    model = PatchTST(n_features=29, seq_len=252, horizon=5, band_mode="direct")
    x = torch.randn(4, 252, 29)
    out = model(x)
    assert out.line.shape == (4, 5)
    assert out.lower_band.shape == (4, 5)
    assert out.upper_band.shape == (4, 5)

def test_patchtst_revin_roundtrip():
    """RevIN normalize·denormalize가 학습 grad에 영향 안 주는지 확인"""
    ...

def test_patchtst_channel_independence():
    """채널 순서 셔플해도 출력 유사 (CI는 채널 순서 불변)"""
    ...
```

---

### 3단계 — CNN-LSTM 안정화 (C-2)

#### 3.1 `ai/models/blocks.py` 신규 (공용 컴포넌트)

```python
class AttentionPooling1D(nn.Module):
    """LSTM 출력 [B, L, H]에 attention 기반 weighted average."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x):  # [B, L, H]
        scores = self.attn(x)                # [B, L, 1]
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)      # [B, H]
```

#### 3.2 `ai/models/cnn_lstm.py` 재구현

```python
class CNNLSTM(MultiHeadForecastModel):
    def __init__(
        self,
        n_features: int = 29,
        seq_len: int = 252,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        n_layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.2,
        band_mode: str = "direct",
    ):
        super().__init__(hidden_dim=lstm_hidden, horizon=horizon, band_mode=band_mode)
        # Conv block with LayerNorm + residual
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(cnn_channels)  # along channel dim after permute
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(cnn_channels)
        self.conv_residual_proj = nn.Conv1d(n_features, cnn_channels, kernel_size=1)  # residual 차원 맞추기
        self.conv_dropout = nn.Dropout(dropout)
        # LSTM
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        # Attention pooling
        self.attn_pool = AttentionPooling1D(lstm_hidden)
        self.output_dropout = nn.Dropout(dropout)
        self.apply(_init_weights)

    def forward(self, x):  # x: [B, L, C]
        # Conv block
        x_t = x.permute(0, 2, 1)                            # [B, C, L]
        residual = self.conv_residual_proj(x_t)
        h = F.relu(self.conv1(x_t))
        h = self.norm1(h.permute(0, 2, 1)).permute(0, 2, 1)  # LayerNorm은 channel last에서
        h = self.conv_dropout(h)
        h = F.relu(self.conv2(h))
        h = self.norm2(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = h + residual                                    # residual
        h = self.conv_dropout(h)
        # LSTM
        h = h.permute(0, 2, 1)                              # [B, L, cnn_channels]
        lstm_out, _ = self.lstm(h)                          # [B, L, lstm_hidden]
        lstm_out = self.lstm_norm(lstm_out)
        # Attention pooling
        pooled = self.attn_pool(lstm_out)                   # [B, lstm_hidden]
        pooled = self.output_dropout(pooled)
        return self.build_output(pooled)
```

#### 3.3 단위 테스트

```python
def test_cnn_lstm_forward_shape():
    model = CNNLSTM(n_features=29, seq_len=252, horizon=5, band_mode="direct")
    x = torch.randn(4, 252, 29)
    out = model(x)
    assert out.line.shape == (4, 5)

def test_cnn_lstm_attention_pooling_weights_sum_to_one():
    """attention weight가 softmax이므로 합 1"""
    ...
```

---

### 4단계 — TiDE 재구현 (D-1)

#### 4.1 `ai/models/blocks.py` 추가

```python
class ResidualBlock(nn.Module):
    """TiDE-style ResNet block: Linear → ReLU → Dropout → Linear → LayerNorm + residual."""
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.2):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(x + h)
```

#### 4.2 `ai/models/tide.py` 재구현

```python
class TiDE(MultiHeadForecastModel):
    """
    TiDE 골격 (Das et al., 2023, 논문 fidelity 70%).

    포함: Feature projection → Dense Encoder (ResidualBlock × N) →
          Dense Decoder (ResidualBlock × M) → Temporal Decoder → Lookback skip.
    생략 (의도): 미래 covariate 처리 — Lens 데이터에 미래 covariate 없음.
    """
    def __init__(
        self,
        n_features: int = 29,
        seq_len: int = 252,
        feature_dim: int = 16,         # per-timestep feature projection 차원
        enc_dim: int = 256,
        dec_dim: int = 128,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.2,
        band_mode: str = "direct",
        lookback_baseline_idx: int = 0,  # 0 = 첫 채널 (log_return) 기본
    ):
        super().__init__(hidden_dim=dec_dim, horizon=horizon, band_mode=band_mode)
        self.seq_len = seq_len
        self.lookback_baseline_idx = lookback_baseline_idx

        # 1. Feature projection (per-timestep)
        self.feature_proj = nn.Linear(n_features, feature_dim)
        # 2. Dense Encoder
        proj_input_dim = seq_len * feature_dim
        self.enc_input_proj = nn.Linear(proj_input_dim, enc_dim)
        self.encoder = nn.Sequential(*[
            ResidualBlock(enc_dim, dropout=dropout) for _ in range(n_enc_layers)
        ])
        # 3. Dense Decoder
        self.dec_input_proj = nn.Linear(enc_dim, dec_dim * horizon)
        self.decoder = nn.Sequential(*[
            ResidualBlock(dec_dim, dropout=dropout) for _ in range(n_dec_layers)
        ])
        # 4. Temporal Decoder (per-horizon-step)
        self.temporal_decoder = nn.Linear(dec_dim, dec_dim)
        # 5. Lookback skip
        self.lookback_skip = nn.Linear(seq_len, horizon)

        self.apply(_init_weights)

    def forward(self, x):  # x: [B, L, C]
        B, L, C = x.shape
        # 1. Feature projection
        x_proj = self.feature_proj(x)                   # [B, L, feature_dim]
        # 2. Encoder
        h = x_proj.flatten(start_dim=1)                  # [B, L * feature_dim]
        h = self.enc_input_proj(h)                       # [B, enc_dim]
        h = self.encoder(h)                              # [B, enc_dim]
        # 3. Decoder
        h = self.dec_input_proj(h)                       # [B, dec_dim * horizon]
        h = h.view(B, self.horizon, -1)                  # [B, horizon, dec_dim]
        h = self.decoder(h)                              # [B, horizon, dec_dim]
        # 4. Temporal Decoder
        h = self.temporal_decoder(h)                     # [B, horizon, dec_dim]
        # build_output 준비: pool to [B, dec_dim] (head expects single hidden)
        # 또는 head를 horizon-aware로 바꿈. 여기선 단순화: horizon 평균
        h_pooled = h.mean(dim=1)                         # [B, dec_dim]
        out = self.build_output(h_pooled)
        # 5. Lookback skip
        baseline_series = x[..., self.lookback_baseline_idx]  # [B, L]
        skip = self.lookback_skip(baseline_series)            # [B, horizon]
        # line·band 모두에 skip 더함 (트렌드 baseline)
        return ForecastOutput(
            line=out.line + skip,
            lower_band=out.lower_band + skip,
            upper_band=out.upper_band + skip,
        )
```

**구현 메모**:
- `n_features=29`, `seq_len=252`, `feature_dim=16` → enc input = 252×16 = 4032 → enc_dim=256.
- `n_enc_layers=4`, `n_dec_layers=2` 초안. sweep 단계에서 1~2 step 시도.
- Lookback skip는 첫 채널(log_return) baseline 사용 가정. config로 noteworthy.

#### 4.3 단위 테스트

```python
def test_tide_forward_shape():
    model = TiDE(n_features=29, seq_len=252, horizon=5, band_mode="direct")
    x = torch.randn(4, 252, 29)
    out = model(x)
    assert out.line.shape == (4, 5)

def test_tide_residual_block_preserves_dim():
    block = ResidualBlock(dim=128, dropout=0.2)
    x = torch.randn(4, 128)
    y = block(x)
    assert y.shape == x.shape

def test_tide_lookback_skip_contributes():
    """skip 없이 vs 있을 때 출력 다른지 확인"""
    ...
```

---

### 5단계 — Init 통일 (E-1)

`ai/models/blocks.py`에 공용 함수:

```python
def _init_weights(module):
    """BERT 식 truncated normal init (std=0.02)."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.trunc_normal_(param, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)
```

각 모델 `__init__` 끝에 `self.apply(_init_weights)` 호출. 이미 §2.2, §3.2, §4.2에 반영됨.

---

### 6단계 — Dropout 위치 표준화 (E-2)

§2~§4 모델 코드에 이미 반영:

| 모델 | 추가된 dropout 위치 |
|---|---|
| PatchTST | `input_dropout` (position embedding 직후), `output_dropout` (encoder 출력 직후). TransformerEncoderLayer 내부 dropout은 PyTorch 기본 유지. |
| CNN-LSTM | `conv_dropout` (Conv 블록 후), `output_dropout` (attention pooling 직후). LSTM 사이 dropout은 `n_layers > 1`일 때 PyTorch 기본 유지. |
| TiDE | `ResidualBlock` 내부 dropout (각 block마다). Encoder/decoder 사이는 ResidualBlock 자체가 norm + residual을 갖고 있어 추가 불필요. |

각 모델은 `dropout` 단일 인자를 받아 모든 위치에 동일 비율 적용. sweep 시 비율만 조정.

---

### 7단계 — Dry-run 통합 검증

`ai/train.py`의 dry-run 모드에 모델별 forward 한 batch 테스트 추가:

```bash
uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5 --model patchtst
uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5 --model cnn_lstm
uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5 --model tide
uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5 --model patchtst --band-mode param
```

기대 동작:
- 각 모델이 forward 한 번 통과 (OOM·shape mismatch 없음)
- Loss 한 번 계산 (NaN/Inf 없음)
- `apply_band_postprocess` 호출 후 lower ≤ upper 검증 print
- 종료

---

## 테스트 필수 항목

### 8.1 신규 테스트

1. **출력 헤드 정합성** (P1-1, P1-2):
   - `test_postprocess_preserves_line_head`: 후처리 입력 line과 출력 line이 bit-identical.
   - `test_postprocess_band_sorted`: band가 lower ≤ upper.
   - `test_train_val_inference_postprocess_identical`: 동일 입력에 train val 함수 vs inference 함수 출력 일치.

2. **Width loss** (P2-3):
   - `test_width_loss_nonnegative`: 음수 폭에서도 loss ≥ 0.

3. **출력 파라미터화** (A-3):
   - `test_band_mode_param_lower_le_upper`: `band_mode="param"`에서 lower ≤ upper 항상 성립 (1000 random samples).

4. **PatchTST** (B-2):
   - `test_patchtst_forward_shape`
   - `test_patchtst_revin_normalization`: RevIN 후 batch 평균 ≈ 0, std ≈ 1.
   - `test_patchtst_channel_independent_shape`: CI=True일 때 forward shape 정상.

5. **CNN-LSTM** (C-2):
   - `test_cnn_lstm_forward_shape`
   - `test_cnn_lstm_attention_pooling_sums_to_one`

6. **TiDE** (D-1):
   - `test_tide_forward_shape`
   - `test_residual_block_preserves_dim`
   - `test_tide_lookback_skip_contributes`

7. **Init 통일** (E-1):
   - `test_init_weights_trunc_normal`: 모든 nn.Linear weight std ≈ 0.02 (10% 허용).

### 8.2 회귀 (기존 25 테스트 green 유지)

- `uv run pytest tests/` 모두 green이어야 종료.

---

## 종료 보고 포맷

```
[CP3.5] 완료

## 1. 출력 정합성 fix
- P1-1 (line head 보존): Y / N
- P1-2 (단일 후처리 함수): `ai/postprocess.py:apply_band_postprocess` Y / N
  - val·test·inference 모두 호출 위치: _
- P2-3 (width loss 음수 방지): Y / N

## 2. 출력 헤드 두 방식 (A-3)
- band_mode="direct" 구현: Y / N
- band_mode="param" 구현 (center + log_half_width): Y / N
- 두 방식 forward shape 동일성 검증: Y / N

## 3. PatchTST (B-2)
**핵심 구성요소 존재 여부 체크리스트**:
- [ ] RevIN normalize/denormalize
- [ ] Channel Independence (입력 reshape [B*C, L, 1])
- [ ] Patching (patch_len, stride)
- [ ] Transformer encoder
- [ ] Learned positional embedding
- [ ] Line head + Band head 분리
- [ ] Init 통일 (trunc_normal std=0.02)
- [ ] Dropout 위치: input dropout + encoder 내부 + output dropout

- 모델 forward smoke (B=4, L=252, C=29) 통과: Y / N
- 메모리 / GPU / batch size 변경 사항: _

## 4. CNN-LSTM (C-2)
**핵심 구성요소 존재 여부 체크리스트**:
- [ ] Conv 블록 (2층 Conv1d)
- [ ] LayerNorm (Conv 후)
- [ ] Residual connection (1x1 Conv proj)
- [ ] LSTM (n_layers=2)
- [ ] LSTM 출력 LayerNorm
- [ ] Attention pooling
- [ ] Init 통일
- [ ] Dropout 위치: conv 후 + attention pooling 후 + LSTM 내부

- 모델 forward smoke 통과: Y / N

## 5. TiDE (D-1)
**핵심 구성요소 존재 여부 체크리스트**:
- [ ] Feature projection (per-timestep)
- [ ] Dense Encoder (ResidualBlock × n_enc_layers)
- [ ] Dense Decoder (ResidualBlock × n_dec_layers)
- [ ] Temporal Decoder
- [ ] Lookback skip (baseline channel → horizon)
- [ ] Init 통일
- [ ] Dropout 위치: ResidualBlock 내부

**의도된 갭 (논문 대비)**:
- 미래 covariate 처리 미구현: 사유 — Lens 데이터에 미래 covariate 없음 (D-1 결정)

- 모델 forward smoke 통과: Y / N

## 6. Init 통일 (E-1)
- 적용 모델: PatchTST / CNN-LSTM / TiDE 모두 Y / N
- 검증: nn.Linear weight std 평균 ≈ 0.02 (실측: _)

## 7. Dropout 위치 (E-2)
- PatchTST: input + output 위치 추가 Y / N
- CNN-LSTM: conv 후 + attention pooling 후 추가 Y / N
- TiDE: ResidualBlock 통합 Y / N

## 8. 테스트
- 기존 25 테스트 green: Y / N
- 신규 테스트: _건 (이름 리스트)
- 전체 테스트 수: _

## 9. Dry-run 검증
- patchtst direct dry-run: Y / N
- patchtst param dry-run: Y / N
- cnn_lstm direct dry-run: Y / N
- tide direct dry-run: Y / N
- 각 모델 lower ≤ upper 출력 검증: Y / N

## 10. 회귀 / 빌더 영향
- ai/train.py 영향: _ (인자 추가, 기본값, dry-run 동작)
- ai/inference.py 영향: _ (postprocess 호출 위치)
- ai/loss.py 영향: _ (band_mode 분기)

## 11. CP4 준비 상태
- ticker embedding 인터페이스 자리 확보 (모델별 ticker_id 인자 받을 위치): Y / N
- model_architecture.md 결정 6건 (A-3, B-2, C-2, D-1, E-1, E-2) 모두 코드 반영: Y / N

## 12. 메모 / 의도된 갭 / 다음 단계 권고
- (TiDE D-1 미래 covariate 미구현 등 갭 재명시)
- (메모리·속도 이슈 발견 시 기록)
- (CP4에서 처리할 항목)
```

---

## 체크포인트 준수

- 이 CP는 **단일 종료 보고**. 중간 보고 없음.
- 위 종료 보고 12개 섹션 전부 숫자·Y/N 명시. "추후 확정"·"파일 참고" 금지.
- **핵심 구성요소 체크리스트는 모든 모델에서 필수 채움**. 누락 항목이 있으면 그 자리에 "MISSING — 사유: _" 명시.
- 학습 실제 실행 금지. dry-run·forward smoke만.
- 기존 25 테스트가 깨지면 종료 불가. 모든 테스트 green이 종료 조건.
- CP4 (ticker embedding) 지시서는 본 보고 통과 후 별도 발주.
