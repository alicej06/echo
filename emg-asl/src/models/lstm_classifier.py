"""LSTM-based ASL classifier for sEMG sequences."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import (
    WINDOW_SAMPLES, N_CHANNELS, N_CLASSES,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
)


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for sEMG sequence classification.

    Input:  (batch, seq_len, n_channels)  e.g. (B, 40, 8)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "LSTMClassifier":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
