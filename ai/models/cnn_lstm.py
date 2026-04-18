"""
CNN-LSTM 하이브리드 모델
- CNN: 로컬 패턴 추출
- LSTM: 시계열 의존성 학습

입력: (batch, seq_len, n_features) = (B, 60, 17)
출력: (batch, n_horizons) = (B, 4)
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 17,
        seq_len: int = 60,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        n_layers: int = 2,
        n_outputs: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        x = x.permute(0, 2, 1)          # (B, n_features, seq_len)
        x = self.conv(x)                 # (B, cnn_channels, seq_len)
        x = x.permute(0, 2, 1)          # (B, seq_len, cnn_channels)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])             # (B, n_outputs)
        return torch.sigmoid(out)
