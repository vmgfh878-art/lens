# CP4 지시서 — Ticker embedding 도입 + RevIN denorm 정착 + CI 출력 정착 + Init 테스트 정제

## 배경

CP3.5에서 출력 정합성과 모델 fidelity는 닫았으나 검수 라운드에서 **F-1·F-2·F-3 후속 항목 3건**이 도출됐다. 사용자 판단:
- **F-1 (RevIN denormalize 미사용)**: RevIN은 normalize → 모델 → **denormalize**가 본체. 반쪽 구현은 의미 없음. 제대로 고친다.
- **F-2 (CI 출력 channel-wise 평균)**: target channel 명시 없이 29개 채널 출력을 평균하는 건 CI 의도와 어긋남. target channel 기반으로 정착.
- **F-3 (init 테스트 tolerance 15-25%)**: 표준편차 1~2% 이내가 정상. 측정 방식 정제.

이 셋을 CP4 (Ticker embedding) 본 작업과 한 묶음으로 처리한다. CP4 본 작업은 학습 대상 ticker (1D 473 / 1W 421)에 대한 embedding을 세 모델에 추가하고 dataset · train · inference 경로에 통합하는 것.

근거 문서:
- `docs/model_architecture.md` — F-1·F-2 검토 (CP3.5 검수 라운드 D23).
- `docs/training_hyperparameters.md` — ticker embedding 차원 결정 가능 상태 (1D=473, 1W=421).
- `docs/project_journal.md` — D23 (sweep ablation 권고 항목으로 등재됐으나 사용자 결정으로 본 CP에 흡수).

---

## 목표

### CP4 본 작업 (Ticker embedding)

1. **Ticker → int ID 매핑** 정착: 1D / 1W 별도 매핑 (학습 대상이 다름). 결정론적 (알파벳 정렬) + 직렬화.
2. **세 모델에 `nn.Embedding` 추가**: PatchTST / CNN-LSTM / TiDE 모두.
3. **Embedding 결합 방식 결정 + 구현**: concat-after-backbone 권고 (아래 §3 참조).
4. **Dataset / DataLoader가 `ticker_id` 동반 yield**.
5. **`train.py` · `inference.py` ticker_id 전파**.
6. **추론 시 미등록 ticker는 `unknown_ticker_id` (= num_tickers) 사용**: OOV 방지.

### F-1 (RevIN denormalize 정착)

7. RevIN의 `denormalize`를 **target channel 기준**으로 호출하도록 PatchTST 통합.
8. `target_channel_idx` 인자를 PatchTST init에 추가. 기본값 = `0` (log_return).

### F-2 (CI 출력 정착)

9. PatchTST CI 경로의 출력 결합 방식을 인자로 분기:
   - `ci_aggregate="target"` (기본, 권고): target channel의 예측만 사용.
   - `ci_aggregate="mean"`: 현재 방식 (29개 채널 출력 평균). 호환용 보존.
   - `ci_aggregate="attention"` (옵션): 채널별 attention weight로 결합. 학습 가능 가중치.

### F-3 (Init 테스트 정제)

10. `test_init_weights_trunc_normal`에서 측정 방식을 large tensor 모음으로 변경하고 tolerance를 ±5%로 조인다.

---

## 예상 시간

| 단계 | 시간 |
|---|---|
| Ticker → int ID 매핑 (1D/1W 별도, 직렬화, 로더) | 0.5~1시간 |
| F-1: RevIN denorm 통합 (target channel 기반) | 1.5~2시간 |
| F-2: CI 출력 인자화 (`ci_aggregate`) | 1.5~2시간 |
| F-3: init 테스트 정제 | 0.2~0.5시간 |
| 세 모델에 `nn.Embedding` + concat 결합 | 2.5~3.5시간 |
| Dataset / DataLoader에 ticker_id 동반 | 1~1.5시간 |
| train.py / inference.py 전파 | 1~1.5시간 |
| 테스트 작성·실행 | 2~2.5시간 |
| Dry-run (모델 × 모드 × timeframe 매트릭스) | 0.5~1시간 |
| 보고서 작성 | 0.5시간 |
| **총 체감** | **11~16시간** |

기준: CPU 위주, 풀 학습 미실행. 5060 Ti 16GB 환경에서 dry-run smoke만 수행.

---

## 권한 번들 (사전 승인)

**파일 쓰기 범위**:
- `ai/models/patchtst.py`, `ai/models/cnn_lstm.py`, `ai/models/tide.py`, `ai/models/common.py`
- `ai/models/revin.py` (denorm 활용 정합성 보강)
- `ai/models/blocks.py` (ChannelAttentionPooling 추가 시)
- `ai/datasets.py` 또는 `ai/preprocessing.py` (ticker_id yield)
- `ai/ticker_registry.py` (신규 — 매핑 로더·세이버)
- `ai/cache/ticker_id_map_1d.json`, `ai/cache/ticker_id_map_1w.json` (생성)
- `ai/train.py`, `ai/inference.py`
- `tests/**`
- `docs/cp4_report.md` (종료 리포트)

**허용 명령**:
- `uv run pytest tests/`
- `uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5 --model patchtst`
- `uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5 --model patchtst --ci-aggregate mean`
- `uv run python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5 --model patchtst --ci-aggregate attention`
- `uv run python -m ai.train --dry-run --timeframe 1W --seq-len 104 --horizon 4 --model patchtst`
- `uv run python -m ai.ticker_registry --build` (매핑 생성 1회)
- `uv run python -c "..."` (forward 형상 임시 확인)

**금지**:
- 학습 실제 실행 금지 (1 epoch 이상)
- DB 쓰기 금지
- `git push` 금지
- 기존 41 테스트 빨간색 만드는 회귀 금지

**휴먼 개입 트리거**:
- Embedding 도입으로 모델 forward shape 깨짐, 30분 이상 디버그
- F-1 적용 후 PatchTST 출력 분포가 명백히 비정상 (NaN, 극단값)
- 매핑 파일 직렬화 형식 결정 애매 (json vs parquet vs csv) — 권고 json (가독성·재현성)
- 예상 시간 2배 초과 (24시간+)

---

## 실행 상세

### 1단계 — Ticker ID 매핑 정착

#### 1.1 `ai/ticker_registry.py` 신규

```python
"""
Ticker → int ID 매핑.

원칙:
- timeframe별 별도 매핑 (1D 473 / 1W 421 학습 대상이 다르기 때문).
- 결정론적 (알파벳 오름차순). 재현 가능.
- 직렬화: json (가독성·재현성).
- OOV 처리: 등록되지 않은 ticker는 unknown_ticker_id = num_tickers.
"""
from __future__ import annotations
import json
from pathlib import Path

CACHE_DIR = Path("ai/cache")
DEFAULT_PATH = {
    "1D": CACHE_DIR / "ticker_id_map_1d.json",
    "1W": CACHE_DIR / "ticker_id_map_1w.json",
}

def build_registry(eligible_tickers: list[str], timeframe: str) -> dict[str, int]:
    sorted_tickers = sorted(set(eligible_tickers))
    mapping = {ticker: idx for idx, ticker in enumerate(sorted_tickers)}
    return mapping

def save_registry(mapping: dict[str, int], timeframe: str, path: Path | None = None) -> Path:
    target = path or DEFAULT_PATH[timeframe]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump({"timeframe": timeframe, "mapping": mapping, "num_tickers": len(mapping)}, f, indent=2, ensure_ascii=False)
    return target

def load_registry(timeframe: str, path: Path | None = None) -> dict:
    target = path or DEFAULT_PATH[timeframe]
    with target.open(encoding="utf-8") as f:
        return json.load(f)

def lookup_id(ticker: str, registry: dict) -> int:
    """OOV는 unknown_ticker_id (= num_tickers)."""
    mapping = registry["mapping"]
    return mapping.get(ticker, registry["num_tickers"])
```

#### 1.2 매핑 빌드 진입점

```bash
uv run python -m ai.ticker_registry --build
```

내부 로직:
1. `build_dataset_plan(timeframe="1D")` 호출 → eligible_tickers 1D 473.
2. `build_registry(eligible_tickers, "1D")` → 직렬화.
3. 1W 동일 (eligible_tickers 421).
4. 출력: 두 파일 경로 + ticker 수.

테스트:
- `test_ticker_registry_alphabetical`: 알파벳 정렬 후 ID가 0부터 시작하는지.
- `test_ticker_registry_oov`: 등록 안 된 ticker가 num_tickers 반환하는지.
- `test_ticker_registry_roundtrip`: save → load → lookup 일치.

---

### 2단계 — F-1: RevIN denormalize 정착

#### 2.1 `ai/models/revin.py` 보강

`denormalize_target` 메서드 추가:

```python
class RevIN(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        # ... 기존 코드 그대로 ...
        # gamma·beta는 [1, 1, n_features] 형태

    def denormalize_target(self, y: torch.Tensor, target_channel_idx: int) -> torch.Tensor:
        """
        타깃 채널 기준 denormalize.

        y: 모델 출력 [B, H] (line, lower, upper 각각)
        target_channel_idx: 입력 채널 중 타깃에 해당하는 인덱스 (예: log_return = 0)

        역순:
          1. affine 역변환: y = (y - beta_target) / (gamma_target + eps)
          2. 표준화 역변환: y = y * std_target + mean_target
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("denormalize_target 호출 전 normalize가 먼저 실행돼야 함.")
        mean_target = self._mean[..., target_channel_idx]   # [B, 1]
        std_target = self._std[..., target_channel_idx]     # [B, 1]
        out = y
        if self.affine:
            beta_t = self.beta[..., target_channel_idx]      # scalar (or [1, 1])
            gamma_t = self.gamma[..., target_channel_idx]    # scalar (or [1, 1])
            out = (out - beta_t) / (gamma_t + self.eps)
        # mean_target [B, 1] broadcasts to [B, H]
        return out * std_target + mean_target
```

#### 2.2 PatchTST forward에 denorm 호출 통합

```python
class PatchTST(MultiHeadForecastModel):
    def __init__(
        self,
        ...,
        target_channel_idx: int = 0,  # 신규
        ci_aggregate: str = "target",  # F-2: 신규
        ...
    ):
        ...
        self.target_channel_idx = target_channel_idx
        self.ci_aggregate = ci_aggregate
        ...

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> ForecastOutput:
        B, L, C = x.shape
        if self.use_revin:
            x = self.revin(x, mode="norm")

        # ... encoder 처리 ...
        # 결과: per_channel line, lower, upper (CI 경로)
        #   또는 single line, lower, upper (non-CI 경로)

        # F-2: CI aggregate 분기
        if self.channel_independent:
            line_per_ch = ...  # [B, C, H]
            lower_per_ch = ...
            upper_per_ch = ...
            line_out, lower_out, upper_out = self._aggregate_channels(
                line_per_ch, lower_per_ch, upper_per_ch
            )
        else:
            line_out, lower_out, upper_out = ... # non-CI 경로 직접 출력

        # F-1: RevIN denormalize (target channel 기반)
        if self.use_revin:
            line_out = self.revin.denormalize_target(line_out, self.target_channel_idx)
            lower_out = self.revin.denormalize_target(lower_out, self.target_channel_idx)
            upper_out = self.revin.denormalize_target(upper_out, self.target_channel_idx)

        # Ticker embedding (CP4 본 작업) — §4에서 정의
        # 이 단계에선 자리만 표시:
        # if ticker_id is not None: line_out, lower_out, upper_out 보정

        return ForecastOutput(line=line_out, lower_band=lower_out, upper_band=upper_out)
```

#### 2.3 RevIN denorm 검증 테스트

- `test_revin_normalize_denormalize_roundtrip`: 입력 채널 i를 normalize → denormalize_target(i) 했을 때 원값 복원 (수치 오차 ≤ eps).
- `test_revin_target_channel_selection`: target_channel_idx에 따라 denorm 결과가 달라지는지 (서로 다른 채널이면 다른 mean/std 사용).

---

### 3단계 — F-2: CI 출력 정착 (`ci_aggregate` 분기)

#### 3.1 세 가지 모드 구현

PatchTST `_aggregate_channels` 메서드:

```python
def _aggregate_channels(
    self,
    line_per_ch: torch.Tensor,    # [B, C, H]
    lower_per_ch: torch.Tensor,
    upper_per_ch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.ci_aggregate == "target":
        idx = self.target_channel_idx
        return line_per_ch[:, idx], lower_per_ch[:, idx], upper_per_ch[:, idx]
    elif self.ci_aggregate == "mean":
        return line_per_ch.mean(dim=1), lower_per_ch.mean(dim=1), upper_per_ch.mean(dim=1)
    elif self.ci_aggregate == "attention":
        # 학습 가능 채널 attention. ChannelAttentionPooling 모듈 사용.
        return self.channel_attn(line_per_ch), self.channel_attn(lower_per_ch), self.channel_attn(upper_per_ch)
    else:
        raise ValueError(f"unknown ci_aggregate: {self.ci_aggregate}")
```

#### 3.2 `ai/models/blocks.py`에 `ChannelAttentionPooling` 추가

```python
class ChannelAttentionPooling(nn.Module):
    """채널별 출력 [B, C, H]를 학습 가능 attention으로 가중 평균."""
    def __init__(self, n_channels: int):
        super().__init__()
        self.attn = nn.Linear(n_channels, n_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H]
        # 채널별 score: [B, C]
        # H 평균 후 attention
        score = self.attn(x.mean(dim=-1))  # [B, C]
        weight = torch.softmax(score, dim=-1).unsqueeze(-1)  # [B, C, 1]
        return (x * weight).sum(dim=1)  # [B, H]
```

PatchTST에서 `ci_aggregate="attention"`일 때만 `self.channel_attn = ChannelAttentionPooling(n_features)` 인스턴스화.

#### 3.3 CI aggregate 테스트

- `test_ci_aggregate_target_uses_target_channel`: `ci_aggregate="target"`일 때 출력 = target channel 예측.
- `test_ci_aggregate_mean_matches_legacy`: `ci_aggregate="mean"`일 때 출력 = 채널 평균 (CP3.5 호환).
- `test_ci_aggregate_attention_weights_sum_to_one`: attention 가중치 합 = 1.

---

### 4단계 — Ticker Embedding 통합 (CP4 본 작업)

#### 4.1 결합 방식 결정: **concat-after-backbone**

세 가지 옵션 비교:

| 방식 | 위치 | 장점 | 단점 |
|---|---|---|---|
| **(권고) concat-after-backbone** | encoder/backbone 출력 hidden에 ticker_emb concat → head | 단순. ticker 정보가 head로만 전달 (encoder는 ticker 무관 학습). 해석 명확. | encoder는 ticker 정보 못 봄. |
| add-to-input | 입력 [B, L, C]에 ticker_emb broadcast 후 더함 | encoder가 ticker별 패턴 학습 가능. | 표현력 제한 (단순 합). |
| FiLM | hidden = γ(ticker) * hidden + β(ticker) | 가장 표현력 강함. | 파라미터 더 많음. 디버그 복잡. |

**권고 = concat-after-backbone**: Phase 1 단계에서 단순·해석성 우선. FiLM은 Phase 2 옵션.

#### 4.2 Embedding 차원

- 1D: 473 ticker → embedding 행렬 (473+1) × 32 = 15,168 params (negligible).
- 1W: 421 ticker → (421+1) × 32 = 13,504 params.
- `+1` = unknown_ticker_id (OOV).
- 차원 32: 일반적인 categorical embedding 권장 범위 (sqrt(num_categories) ≈ 21 ~ 32).

#### 4.3 모델별 통합

##### PatchTST

```python
class PatchTST(MultiHeadForecastModel):
    def __init__(
        self,
        ...,
        num_tickers: int = 0,   # 0이면 embedding 비활성
        ticker_emb_dim: int = 32,
    ):
        # backbone hidden_dim 변경됨: d_model * n_patches + ticker_emb_dim
        self.use_ticker_emb = num_tickers > 0
        ticker_extra = ticker_emb_dim if self.use_ticker_emb else 0
        super().__init__(
            hidden_dim=d_model * self.n_patches + ticker_extra,
            horizon=horizon, band_mode=band_mode,
        )
        if self.use_ticker_emb:
            self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim)  # +1 OOV
        ...

    def forward(self, x, ticker_id=None):
        # ... encoder + CI aggregate + revin denorm까지 진행 ...
        # backbone에서 만든 flattened hidden [B, d_model * n_patches]
        # 단, CI 경로에서는 build_output 호출 전에 hidden을 받아오는 구조 변경 필요.
        # → CI 경로에서도 채널별 hidden 평균/타깃 후 [B, d_model * n_patches] 만들고
        #   ticker_emb concat → head.
        if self.use_ticker_emb:
            assert ticker_id is not None
            ticker_emb = self.ticker_embedding(ticker_id)   # [B, ticker_emb_dim]
            hidden_with_ticker = torch.cat([hidden, ticker_emb], dim=-1)
        else:
            hidden_with_ticker = hidden
        return self.build_output(hidden_with_ticker)
```

**구현 메모**: PatchTST CI 경로는 채널별 hidden을 head 통과 후 aggregate 했었음. 이제 aggregate를 hidden 단계에서 한 후 ticker concat → head로 순서 변경 필요. 즉 `_aggregate_channels`가 hidden을 다루도록 시그니처 변경.

##### CNN-LSTM

```python
class CNNLSTM(MultiHeadForecastModel):
    def __init__(
        self,
        ...,
        num_tickers: int = 0,
        ticker_emb_dim: int = 32,
    ):
        ticker_extra = ticker_emb_dim if num_tickers > 0 else 0
        super().__init__(
            hidden_dim=lstm_hidden + ticker_extra,
            horizon=horizon, band_mode=band_mode,
        )
        self.use_ticker_emb = num_tickers > 0
        if self.use_ticker_emb:
            self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim)
        ...

    def forward(self, x, ticker_id=None):
        # ... conv + lstm + attention pooling → pooled [B, lstm_hidden]
        if self.use_ticker_emb:
            ticker_emb = self.ticker_embedding(ticker_id)
            pooled = torch.cat([pooled, ticker_emb], dim=-1)
        return self.build_output(pooled)
```

##### TiDE

```python
class TiDE(MultiHeadForecastModel):
    def __init__(
        self,
        ...,
        num_tickers: int = 0,
        ticker_emb_dim: int = 32,
    ):
        ticker_extra = ticker_emb_dim if num_tickers > 0 else 0
        super().__init__(
            hidden_dim=dec_dim + ticker_extra,
            horizon=horizon, band_mode=band_mode,
        )
        self.use_ticker_emb = num_tickers > 0
        if self.use_ticker_emb:
            self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim)
        ...

    def forward(self, x, ticker_id=None):
        # ... encoder + decoder + temporal_decoder + pool → pooled [B, dec_dim]
        if self.use_ticker_emb:
            ticker_emb = self.ticker_embedding(ticker_id)
            pooled = torch.cat([pooled, ticker_emb], dim=-1)
        out = self.build_output(pooled)
        # lookback_skip 더하기는 그대로
        ...
```

#### 4.4 Dataset / DataLoader 변경

`ai/preprocessing.py` 또는 `ai/datasets.py`:

```python
class ForecastDataset(Dataset):
    def __init__(self, samples, ticker_registry, ...):
        self.samples = samples  # 각 sample = (ticker, x, y_line, y_band, ...)
        self.ticker_registry = ticker_registry

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ticker_id = lookup_id(sample["ticker"], self.ticker_registry)
        return {
            "x": sample["x"],
            "y_line": sample["y_line"],
            "y_band": sample["y_band"],
            "ticker_id": torch.tensor(ticker_id, dtype=torch.long),
        }
```

DataLoader collate가 자동으로 stack — `ticker_id`는 [B] LongTensor.

#### 4.5 train.py / inference.py 전파

```python
# train.py 학습 루프
for batch in dataloader:
    x, y_line, y_band, ticker_id = batch["x"], batch["y_line"], batch["y_band"], batch["ticker_id"]
    output = model(x, ticker_id=ticker_id)
    loss = composite_loss(output, y_line, y_band)
    ...

# inference.py
for ticker, x in inference_inputs:
    ticker_id = lookup_id(ticker, registry)
    ticker_id_tensor = torch.tensor([ticker_id], dtype=torch.long)
    output = model(x.unsqueeze(0), ticker_id=ticker_id_tensor)
    line, lower, upper = apply_band_postprocess(output.line, output.lower_band, output.upper_band)
    ...
```

#### 4.6 CLI 인자 추가

`ai/train.py`:
- `--num-tickers` (auto: registry에서 읽음)
- `--ticker-emb-dim` (기본 32)
- `--ci-aggregate` (target / mean / attention, 기본 target)

---

### 5단계 — F-3: Init 테스트 정제

#### 5.1 측정 방식 변경

기존 (느슨):
```python
# 단일 작은 Linear에서 std 측정 → noise 큼
weight = model.line_head.weight.flatten()
std = weight.std()
assert 0.017 < std < 0.025  # 15-25% tolerance
```

변경 (엄밀):
```python
# 모델 내 모든 nn.Linear weight를 모아서 측정
all_weights = []
for m in model.modules():
    if isinstance(m, nn.Linear) and m.weight.numel() >= 1024:  # 충분히 큰 layer만
        all_weights.append(m.weight.flatten())
combined = torch.cat(all_weights)
std = combined.std().item()
mean = combined.mean().item()
assert abs(std - 0.02) / 0.02 < 0.05, f"std deviation {std} > 5% from 0.02"
assert abs(mean) < 0.005, f"mean {mean} not near 0"
```

5% tolerance + mean 검증 추가.

#### 5.2 모델별 적용

- `test_init_weights_trunc_normal_patchtst`
- `test_init_weights_trunc_normal_cnn_lstm`
- `test_init_weights_trunc_normal_tide`

각각 큰 모델 인스턴스화 후 측정.

---

### 6단계 — Dry-run 통합 검증

매트릭스 dry-run:

```bash
# CI aggregate 3종 × 모델 3종 + 1D/1W
uv run python -m ai.train --dry-run --timeframe 1D --model patchtst --ci-aggregate target
uv run python -m ai.train --dry-run --timeframe 1D --model patchtst --ci-aggregate mean
uv run python -m ai.train --dry-run --timeframe 1D --model patchtst --ci-aggregate attention
uv run python -m ai.train --dry-run --timeframe 1D --model cnn_lstm
uv run python -m ai.train --dry-run --timeframe 1D --model tide
uv run python -m ai.train --dry-run --timeframe 1W --model patchtst --ci-aggregate target

# band_mode param도 확인
uv run python -m ai.train --dry-run --timeframe 1D --model patchtst --band-mode param
```

기대 동작:
- 각 dry-run에서 모델 forward 1회 + loss 1회 + apply_band_postprocess 호출.
- ticker_id가 batch에 포함됐는지 print 확인.
- RevIN denorm 후 출력 분포가 raw target 분포 범위 (log_return ~ ±0.05 정도) 안에 있는지 sanity check.

---

## 테스트 필수 항목

### 신규 테스트 (예상 15+건)

1. **Ticker registry**:
   - `test_ticker_registry_alphabetical`
   - `test_ticker_registry_oov_returns_num_tickers`
   - `test_ticker_registry_save_load_roundtrip`
   - `test_ticker_registry_1d_and_1w_independent`

2. **F-1 RevIN denorm**:
   - `test_revin_normalize_denormalize_roundtrip` (입력 채널 i를 norm→denorm_target(i) 시 원값 복원)
   - `test_revin_target_channel_selection_changes_output`
   - `test_patchtst_revin_full_pipeline_output_in_target_scale` (모델 출력이 raw log_return scale인지 확인)

3. **F-2 CI aggregate**:
   - `test_ci_aggregate_target_equals_target_channel_only`
   - `test_ci_aggregate_mean_matches_legacy_behavior`
   - `test_ci_aggregate_attention_weights_sum_to_one`

4. **Ticker embedding**:
   - `test_patchtst_ticker_embedding_changes_output` (다른 ticker_id면 다른 출력)
   - `test_cnn_lstm_ticker_embedding_changes_output`
   - `test_tide_ticker_embedding_changes_output`
   - `test_ticker_embedding_oov_uses_unknown_id`
   - `test_model_without_ticker_embedding_backward_compat` (`num_tickers=0`이면 ticker_id 무시 동작)

5. **F-3 Init 정제**:
   - `test_init_weights_trunc_normal_patchtst` (5% tolerance)
   - `test_init_weights_trunc_normal_cnn_lstm`
   - `test_init_weights_trunc_normal_tide`

### 회귀 (기존 41 테스트 green)

- `uv run pytest tests/` 모두 green이 종료 조건.

---

## 종료 보고 포맷

```
[CP4] 완료

## 1. Ticker registry
- 1D 매핑 ticker 수: _
- 1W 매핑 ticker 수: _
- OOV unknown_ticker_id: _
- 저장 경로: ai/cache/ticker_id_map_1d.json, ai/cache/ticker_id_map_1w.json
- 직렬화 형식: json (timeframe, mapping, num_tickers)
- Roundtrip 테스트 통과: Y/N

## 2. F-1 RevIN denormalize 정착
- denormalize_target(y, target_channel_idx) 메서드 추가: Y/N
- PatchTST forward에 호출 통합 (line/lower/upper 각각): Y/N
- target_channel_idx 기본값: 0 (log_return)
- Roundtrip 검증 (norm → denorm 원값 복원): Y/N (수치 오차: _)
- 모델 출력 raw target scale 진입 확인 (sanity check 분포): _

## 3. F-2 CI 출력 정착
- ci_aggregate 인자 추가: Y/N
- target / mean / attention 3종 구현: Y/N/Y
- ChannelAttentionPooling 모듈: ai/models/blocks.py 추가 Y/N
- 기본값: target

## 4. Ticker embedding 통합
**모델별 체크리스트**:
### PatchTST
- [ ] num_tickers, ticker_emb_dim 인자 추가
- [ ] nn.Embedding(num_tickers + 1, ticker_emb_dim) 보유 (OOV +1)
- [ ] CI aggregate를 hidden 단계로 이동 (head 통과 전)
- [ ] hidden + ticker_emb concat → head 입력
- [ ] forward에 ticker_id 인자 추가, num_tickers=0이면 무시
- [ ] forward smoke (B=4, L=252, C=29, ticker_id=[B]) 통과

### CNN-LSTM
- [ ] num_tickers, ticker_emb_dim 인자 추가
- [ ] nn.Embedding 보유
- [ ] attention pooling 후 ticker_emb concat
- [ ] forward smoke 통과

### TiDE
- [ ] num_tickers, ticker_emb_dim 인자 추가
- [ ] nn.Embedding 보유
- [ ] temporal_decoder pool 후 ticker_emb concat
- [ ] lookback_skip은 ticker 무관 유지
- [ ] forward smoke 통과

- ticker_emb_dim 결정값: 32
- 임베딩 파라미터 수 (1D): _ / (1W): _

## 5. F-3 Init 테스트 정제
- 측정 방식 변경 (큰 layer 모음): Y/N
- Tolerance 5% 적용: Y/N
- 모델별 std 실측 (1D PatchTST): _
- 모델별 std 실측 (1D CNN-LSTM): _
- 모델별 std 실측 (1D TiDE): _

## 6. Dataset / Train / Inference 전파
- ai/datasets.py (또는 preprocessing.py) ticker_id yield: Y/N
- ai/train.py forward 호출에 ticker_id 전달: Y/N
- ai/inference.py lookup_id + ticker_id 텐서화: Y/N
- CLI 인자 --num-tickers, --ticker-emb-dim, --ci-aggregate 추가: Y/N

## 7. 테스트
- 기존 41 테스트 green: Y/N
- 신규 테스트: _건 (이름 리스트)
- 전체 테스트 수: _

## 8. Dry-run 매트릭스
- patchtst target × 1D: Y/N
- patchtst mean × 1D: Y/N
- patchtst attention × 1D: Y/N
- cnn_lstm × 1D: Y/N
- tide × 1D: Y/N
- patchtst target × 1W: Y/N
- patchtst param mode × 1D: Y/N
- 각 dry-run forward 통과 + ticker_id batch 확인: Y/N

## 9. 통합 영향 정리
- ai/train.py: _
- ai/inference.py: _
- ai/loss.py: _ (필요시)
- ai/datasets.py / preprocessing.py: _

## 10. CP5 준비 상태
- Early stopping 추가 위치 (train.py 검증 루프) 확인: Y/N
- 학습 대상 ticker 수 1D/1W 기준 embedding 차원 결정값 코드 반영: Y/N

## 11. 메모 / 의도된 갭 / 다음 단계 권고
- (TiDE 미래 covariate 미구현 그대로 유지)
- (RevIN denorm 적용 후 모델 출력 분포 변화 발견 시 기록)
- (CI attention 모드 학습 시 channel weight 해석 메모)
- (CP5에서 처리할 항목)
```

---

## 체크포인트 준수

- 이 CP는 **단일 종료 보고**. 중간 보고 없음.
- 위 종료 보고 11개 섹션 전부 숫자·Y/N 명시. "추후 확정"·"파일 참고" 금지.
- **모델별 ticker embedding 체크리스트 필수 (RevIN 누락 같은 fidelity gap 재발 방지 — D21 적용)**.
- **F-1·F-2·F-3을 별도 섹션으로 분리해 보고** (CP3.5 후속 항목 추적성 확보).
- 학습 실제 실행 금지. dry-run·forward smoke만.
- 기존 41 테스트가 깨지면 종료 불가.
- CP5 (early stopping) 지시서는 본 보고 통과 후 별도 발주.
