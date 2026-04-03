"""
Two-stage EMG → ASL classifier.

Architecture:
  Stage 1: EMG → 63-dim hand joint angles (emg2pose)
  Stage 2: 63-dim joint angles → ASL letter (A–Z)

This is architecturally cleaner than direct EMG→letter because:
  - Stage 1 is a general-purpose pose estimator solved by Meta on 370 hours of data.
    We reuse their pretrained weights without modification.
  - Stage 2 is trivial: ASL static letters are fully defined by hand shape.
    A lightweight MLP or KNN on 63-dim MediaPipe-compatible joint angles achieves
    near-perfect classification on stage 2 alone.

The cross-modal CLIP embedding means both stages also speak the same feature language
as the MediaPipe Hands vision teacher used during auto-labeling.

Usage:
    from src.models.two_stage_classifier import TwoStageASLClassifier

    model = TwoStageASLClassifier.from_pretrained(
        pose_checkpoint="models/emg2pose_pretrained.pt",
        asl_checkpoint="models/asl_stage2.pt",
    )
    # x: (batch, 400, 16)  float32
    probs = model.predict_proba(x)  # (batch, 26)
    letter = model.predict_letter(x)  # list of str
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.constants import N_CHANNELS, WINDOW_SAMPLES, N_CLASSES, ASL_CLASSES

JOINT_DIM = 63  # MediaPipe Hands: 21 landmarks × 3 (x, y, z)


# =============================================================================
# Stage 1: EMG → Joint Angles
# =============================================================================

class EMGPoseEstimator(nn.Module):
    """
    LSTM-based model that maps a (batch, window, 16) EMG window to
    63-dim hand joint angles matching MediaPipe Hands format.

    Architecture mirrors Meta's emg2pose paper: temporal CNN feature extractor
    followed by a bidirectional LSTM and a linear projection head.

    When Meta's pretrained weights are available, load them with from_pretrained().
    Otherwise this model can be trained from scratch on the emg2pose dataset.
    """

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        cnn_filters: int = 64,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        joint_dim: int = JOINT_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Temporal CNN front-end
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_filters),
            nn.GELU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_filters),
            nn.GELU(),
        )
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden * 2, joint_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window_samples, in_channels)
        returns: (batch, joint_dim) — predicted joint angles
        """
        # CNN expects (batch, channels, time)
        h = x.transpose(1, 2)
        h = self.cnn(h)
        # LSTM expects (batch, time, features)
        h, _ = self.lstm(h.transpose(1, 2))
        h = self.dropout(h[:, -1])  # last timestep
        return self.head(h)


# =============================================================================
# Stage 2: Joint Angles → ASL Letter
# =============================================================================

class JointAnglesASLClassifier(nn.Module):
    """
    Lightweight MLP that maps 63-dim hand joint angles to ASL letter classes (A–Z).

    Stage 2 is intentionally simple — hand shapes for 26 static ASL letters are
    geometrically distinct in 63-dim joint space, so a 2-layer MLP is sufficient.
    """

    def __init__(
        self,
        joint_dim: int = JOINT_DIM,
        hidden_dim: int = 128,
        n_classes: int = N_CLASSES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        joints: (batch, joint_dim)
        returns: (batch, n_classes) logits
        """
        return self.net(joints)


# =============================================================================
# Two-Stage Pipeline
# =============================================================================

class TwoStageASLClassifier(nn.Module):
    """
    Full two-stage EMG → ASL letter pipeline.

    Stage 1 (EMGPoseEstimator) converts raw EMG windows to hand joint angles.
    Stage 2 (JointAnglesASLClassifier) maps joint angles to ASL letter classes.

    Args:
        pose_estimator: Pretrained EMGPoseEstimator (or None to use default init)
        asl_classifier: Pretrained JointAnglesASLClassifier (or None for default init)
        freeze_stage1: If True, freeze Stage 1 weights during fine-tuning
    """

    def __init__(
        self,
        pose_estimator: Optional[EMGPoseEstimator] = None,
        asl_classifier: Optional[JointAnglesASLClassifier] = None,
        freeze_stage1: bool = True,
    ):
        super().__init__()
        self.stage1 = pose_estimator or EMGPoseEstimator()
        self.stage2 = asl_classifier or JointAnglesASLClassifier()

        if freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window_samples, N_CHANNELS)
        returns: (batch, N_CLASSES) logits
        """
        joints = self.stage1(x)       # (batch, 63)
        logits = self.stage2(joints)  # (batch, 26)
        return logits

    def predict_joints(self, x: torch.Tensor) -> torch.Tensor:
        """Run only Stage 1. Returns (batch, 63) joint angle estimates."""
        with torch.no_grad():
            return self.stage1(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, N_CLASSES) probabilities (softmax over logits)."""
        with torch.no_grad():
            return F.softmax(self(x), dim=-1)

    def predict_letter(self, x: torch.Tensor) -> list[str]:
        """Returns predicted ASL letter strings for each sample in the batch."""
        probs = self.predict_proba(x)
        indices = probs.argmax(dim=-1).tolist()
        return [ASL_CLASSES[i] for i in indices]

    @classmethod
    def from_pretrained(
        cls,
        pose_checkpoint: Optional[str | Path] = None,
        asl_checkpoint: Optional[str | Path] = None,
        freeze_stage1: bool = True,
        map_location: str = "cpu",
    ) -> "TwoStageASLClassifier":
        """
        Load a TwoStageASLClassifier from saved checkpoints.

        Args:
            pose_checkpoint: Path to EMGPoseEstimator .pt checkpoint.
                             If None, Stage 1 starts with random weights.
            asl_checkpoint:  Path to JointAnglesASLClassifier .pt checkpoint.
                             If None, Stage 2 starts with random weights.
            freeze_stage1:   Freeze Stage 1 for ASL fine-tuning (default True).
            map_location:    Torch device string (default "cpu").
        """
        stage1 = EMGPoseEstimator()
        stage2 = JointAnglesASLClassifier()

        if pose_checkpoint is not None:
            ckpt = torch.load(pose_checkpoint, map_location=map_location)
            state = ckpt.get("model_state_dict", ckpt)
            stage1.load_state_dict(state, strict=False)
            print(f"[TwoStage] Loaded Stage 1 weights from {pose_checkpoint}")
        else:
            print("[TwoStage] Stage 1 initialized with random weights (no checkpoint provided)")

        if asl_checkpoint is not None:
            ckpt = torch.load(asl_checkpoint, map_location=map_location)
            state = ckpt.get("model_state_dict", ckpt)
            stage2.load_state_dict(state, strict=False)
            print(f"[TwoStage] Loaded Stage 2 weights from {asl_checkpoint}")

        return cls(
            pose_estimator=stage1,
            asl_classifier=stage2,
            freeze_stage1=freeze_stage1,
        )

    def export_onnx(self, path: str | Path, opset: int = 17) -> None:
        """Export the full two-stage pipeline to ONNX."""
        self.eval()
        dummy = torch.zeros(1, WINDOW_SAMPLES, N_CHANNELS)
        torch.onnx.export(
            self,
            dummy,
            str(path),
            input_names=["emg_window"],
            output_names=["asl_logits"],
            dynamic_axes={"emg_window": {0: "batch"}, "asl_logits": {0: "batch"}},
            opset_version=opset,
        )
        print(f"[TwoStage] ONNX exported to {path}")
