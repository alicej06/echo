"""
Cross-modal CLIP-style embedding for EMG <-> Vision alignment.

Architecture:
  EMGEncoder:    (B, feature_dim) -> 128-dim L2-normalized embedding
  VisionEncoder: (B, 63) landmark -> 128-dim L2-normalized embedding
  InfoNCE contrastive loss to align paired (EMG, vision) samples.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import FEATURE_DIM, N_CLASSES, ASL_CLASSES


class EMGEncoder(nn.Module):
    """Map 80-dim EMG feature vector -> 128-dim L2-normalized embedding."""

    def __init__(self, input_dim: int = FEATURE_DIM, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class VisionEncoder(nn.Module):
    """Map 63-dim hand landmark vector -> 128-dim L2-normalized embedding."""

    def __init__(self, input_dim: int = 63, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CrossModalASL(nn.Module):
    """
    Cross-modal EMG-Vision alignment model.
    Uses learnable temperature (log_temp) for InfoNCE loss.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.emg_encoder = EMGEncoder(embed_dim=embed_dim)
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 100.0)

    def forward(self, emg: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss for a batch of paired (EMG, vision) samples.
        """
        e = self.emg_encoder(emg)        # (B, D)
        v = self.vision_encoder(vision)  # (B, D)
        logits = torch.matmul(e, v.T) / self.temperature  # (B, B)
        targets = torch.arange(len(e), device=e.device)
        loss_e2v = F.cross_entropy(logits, targets)
        loss_v2e = F.cross_entropy(logits.T, targets)
        return (loss_e2v + loss_v2e) / 2

    def build_class_gallery(
        self,
        vision_prototypes: dict,
    ) -> torch.Tensor:
        """
        Build per-class vision embeddings from prototype landmarks.

        Args:
            vision_prototypes: {class_idx: (N, 63) landmark array}

        Returns:
            gallery: (N_CLASSES, embed_dim) normalized embeddings
        """
        gallery = torch.zeros(N_CLASSES, self.emg_encoder.net[-1].out_features)
        with torch.no_grad():
            for cls_idx, lm_array in vision_prototypes.items():
                lm_t = torch.from_numpy(lm_array).float()
                emb = self.vision_encoder(lm_t).mean(dim=0)
                gallery[cls_idx] = F.normalize(emb, dim=0)
        return gallery

    def classify_emg(
        self,
        emg_features: torch.Tensor,
        gallery: torch.Tensor,
    ) -> tuple:
        """
        Classify EMG features via nearest-neighbor in embedding space.

        Returns:
            (pred_classes, confidences)
        """
        with torch.no_grad():
            emg_emb = self.emg_encoder(emg_features)  # (B, D)
            sims = torch.matmul(emg_emb, gallery.T)    # (B, N_CLASSES)
            probs = F.softmax(sims / self.temperature, dim=-1)
            preds = probs.argmax(dim=-1)
            confs = probs.max(dim=-1).values
        return preds, confs

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "CrossModalASL":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
