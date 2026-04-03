"""CNN-LSTM classifier: spatial CNN feature extractor + temporal LSTM."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import N_CHANNELS, N_CLASSES, LSTM_HIDDEN


class CNNLSTMClassifier(nn.Module):
    """
    1-D CNN over channels → LSTM over time.

    Input:  (batch, seq_len, n_channels)  e.g. (B, 40, 8)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        cnn_filters: list = None,
        lstm_hidden: int = LSTM_HIDDEN,
        dropout: float = 0.3,
    ):
        super().__init__()
        cnn_filters = cnn_filters or [32, 64]

        # Permute to (B, channels, seq) for Conv1d
        layers = []
        in_ch = n_channels
        for out_ch in cnn_filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, ch) → (B, ch, seq)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)           # (B, filters, seq)
        x = x.permute(0, 2, 1)    # (B, seq, filters)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)
