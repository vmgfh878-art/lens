from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.blocks import AttentionPooling1D, init_weights
from ai.models.common import ForecastOutput, MultiHeadForecastModel


class CNNLSTM(MultiHeadForecastModel):
    """TCN 스타일 dilated conv와 attention pooling을 포함한 CNN-LSTM."""

    def __init__(
        self,
        n_features: int = 36,
        seq_len: int = 252,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        n_layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.2,
        band_mode: str = "direct",
        num_tickers: int = 0,
        ticker_emb_dim: int = 32,
        use_direction_head: bool = False,
        fp32_modules: str = "none",
    ) -> None:
        del seq_len
        self.use_ticker_embedding = num_tickers > 0
        self.use_direction_head = use_direction_head
        self.fp32_modules = self._parse_fp32_modules(fp32_modules)
        ticker_extra = ticker_emb_dim if self.use_ticker_embedding else 0
        super().__init__(hidden_dim=lstm_hidden + ticker_extra, horizon=horizon, band_mode=band_mode)
        self.dilations = (1, 2, 4, 8)
        conv_layers = []
        norm_layers = []
        in_channels = n_features
        for dilation in self.dilations:
            conv_layers.append(nn.Conv1d(in_channels, cnn_channels, kernel_size=3, padding=dilation, dilation=dilation))
            norm_layers.append(nn.LayerNorm(cnn_channels))
            in_channels = cnn_channels
        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_norms = nn.ModuleList(norm_layers)
        self.conv_residual_proj = nn.Conv1d(n_features, cnn_channels, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            cnn_channels,
            lstm_hidden,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.attn_pool = AttentionPooling1D(lstm_hidden)
        self.output_dropout = nn.Dropout(dropout)
        self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim) if self.use_ticker_embedding else None
        self.direction_head = nn.Linear(lstm_hidden + ticker_extra, horizon) if self.use_direction_head else None
        self.apply(init_weights)

    @staticmethod
    def _parse_fp32_modules(value: str) -> set[str]:
        normalized = (value or "none").strip().lower()
        if normalized == "none":
            return set()
        allowed = {"conv", "lstm", "heads"}
        selected = {item.strip() for item in normalized.split(",") if item.strip()}
        unknown = selected - allowed
        if unknown:
            raise ValueError(f"지원하지 않는 fp32 모듈 옵션입니다: {sorted(unknown)}")
        return selected

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def _should_force_fp32(self, module_name: str, tensor: torch.Tensor) -> bool:
        return bool(self.fp32_modules) and module_name in self.fp32_modules and tensor.is_cuda

    def _forward_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        channel_first = x.permute(0, 2, 1)
        residual = self.conv_residual_proj(channel_first)
        hidden = channel_first
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            hidden = conv(hidden)
            hidden = norm(hidden.permute(0, 2, 1)).permute(0, 2, 1)
            hidden = F.relu(hidden)
            hidden = self.conv_dropout(hidden)
        return self.conv_dropout(hidden + residual)

    def _run_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        if not self._should_force_fp32("conv", x):
            return self._forward_conv_stack(x)
        with torch.autocast(device_type="cuda", enabled=False):
            return self._forward_conv_stack(x.float())

    def _run_lstm_stack(self, sequence_hidden: torch.Tensor) -> torch.Tensor:
        # Windows CUDA 환경에서 cuDNN LSTM 출력이 head와 연결된 뒤 프로세스 종료 시
        # 네이티브 크래시를 일으키는 사례가 있어, CNN-LSTM 경로는 cuDNN을 비활성화한다.
        cudnn_enabled = not sequence_hidden.is_cuda
        if not self._should_force_fp32("lstm", sequence_hidden):
            with torch.backends.cudnn.flags(enabled=cudnn_enabled):
                lstm_out, _ = self.lstm(sequence_hidden)
            return self.lstm_norm(lstm_out)
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.backends.cudnn.flags(enabled=cudnn_enabled):
                lstm_out, _ = self.lstm(sequence_hidden.float())
            return self.lstm_norm(lstm_out)

    def _run_heads(self, pooled: torch.Tensor, ticker_id: torch.Tensor | None) -> ForecastOutput:
        if self.use_ticker_embedding:
            if ticker_id is None:
                raise ValueError("ticker embedding이 활성화된 모델은 ticker_id가 필요합니다.")
            pooled = torch.cat((pooled, self.ticker_embedding(ticker_id)), dim=-1)

        if not self._should_force_fp32("heads", pooled):
            output = self.build_output(pooled)
            if self.direction_head is not None:
                output.direction_logit = self.direction_head(pooled)
            return output

        with torch.autocast(device_type="cuda", enabled=False):
            pooled_fp32 = pooled.float()
            output = self.build_output(pooled_fp32)
            if self.direction_head is not None:
                output.direction_logit = self.direction_head(pooled_fp32)
            return output

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> ForecastOutput:
        hidden = self._run_conv_stack(x)
        sequence_hidden = hidden.permute(0, 2, 1)
        lstm_out = self._run_lstm_stack(sequence_hidden)
        pooled = self.attn_pool(lstm_out)
        pooled = self.output_dropout(pooled)
        return self._run_heads(pooled, ticker_id)
