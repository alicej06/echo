"""Conformer (Conv-Transformer) classifier for sEMG sequences."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import N_CHANNELS, N_CLASSES


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(x)


class _GLUDepthwiseConv(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.norm(x).permute(0, 2, 1)  # (B, D, T)
        x = self.pointwise_in(x)
        x, gate = x.chunk(2, dim=1)
        x = x * torch.sigmoid(gate)  # GLU
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pointwise_out(x)
        x = self.dropout(x).permute(0, 2, 1)  # (B, T, D)
        return residual + x


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ff1 = _FeedForward(d_model, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = _GLUDepthwiseConv(d_model, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = _FeedForward(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x)
        # MHSA with pre-LN
        _x = self.attn_norm(x)
        _x, _ = self.attn(_x, _x, _x)
        x = x + self.attn_drop(_x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class ConformerClassifier(nn.Module):
    """
    Conformer-based sEMG sequence classifier.
    Input:  (batch, seq_len, n_channels)
    Output: (batch, n_classes)
    ~1.5M parameters with d_model=128, n_layers=4.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pe = _SinusoidalPE(d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads=n_heads, kernel_size=kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.embedding(x).permute(0, 2, 1)  # (B, T, d_model)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pool
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)
