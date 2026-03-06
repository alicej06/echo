#!/usr/bin/env python3
"""
Multi-GPU training for EMG-ASL models using DistributedDataParallel (DDP).

Designed for the IYA Nvidia Lab cluster (SLURM). Works on any multi-GPU machine.

Usage examples:

  # Single GPU (development):
  python scripts/train_gpu_ddp.py --model lstm --data-dir data/raw/

  # Multi-GPU on one node (e.g., 4 GPUs):
  torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \\
      --model lstm \\
      --data-dir data/raw/ \\
      --epochs 200 \\
      --batch-size 512

  # Multi-GPU with AMP and gradient accumulation:
  torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \\
      --model lstm \\
      --data-dir data/raw/ \\
      --epochs 200 \\
      --batch-size 512 \\
      --amp \\
      --grad-accum 4

  # Cross-modal training on WLASL (if video + EMG data available):
  torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \\
      --model cross_modal \\
      --data-dir data/raw/ \\
      --video-dir data/wlasl/ \\
      --epochs 100 \\
      --batch-size 256

  # With Weights & Biases logging:
  torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \\
      --model lstm \\
      --data-dir data/raw/ \\
      --epochs 200 \\
      --wandb \\
      --wandb-project maia-emg-asl

  # Submit via SLURM:
  sbatch scripts/slurm/train_lstm.sh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure src.* imports work when invoked from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.utils.constants import (
    ASL_LABELS,
    FEATURE_VECTOR_SIZE,
    NUM_CLASSES,
    WINDOW_SIZE_SAMPLES,
    N_CHANNELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rank %(process)d] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP setup / teardown
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int, torch.device]:
    """Initialize the process group and return (rank, world_size, device).

    Reads RANK, LOCAL_RANK, and WORLD_SIZE from environment variables that
    torchrun populates automatically.  Falls back to single-GPU mode when
    those variables are absent.

    Returns
    -------
    rank:
        Global rank of this process across all nodes.
    world_size:
        Total number of processes in the job.
    device:
        CUDA device assigned to this rank, or CPU when no GPU is available.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        logger.warning("No CUDA device found -- running on CPU.")

    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            "DDP initialized: rank=%d / world_size=%d, device=%s",
            rank, world_size, device,
        )
    else:
        logger.info("Single-process mode, device=%s", device)

    return rank, world_size, device


def cleanup_ddp() -> None:
    """Destroy the distributed process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_lstm_dataset(
    data_dir: str,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Load EMG feature data and return (train_dataset, val_dataset).

    Expects labeled session CSV files under *data_dir* in the standard
    EMG-ASL format (produced by src/data/recorder.py).

    The data are split 80/20 by participant where possible, otherwise by
    window index.

    Parameters
    ----------
    data_dir:
        Directory containing session CSV files.
    seed:
        Random seed for the train/val split.

    Returns
    -------
    (train_dataset, val_dataset)
        TensorDatasets containing (features, label_index) pairs.
        features shape: (N, FEATURE_VECTOR_SIZE)
        label_index shape: (N,) int64
    """
    from src.data.loader import create_windows, load_dataset, extract_features

    df = load_dataset(data_dir)

    # create_windows returns raw EMG windows; extract_features turns them into
    # the 80-dim feature vectors expected by the LSTM.
    windows, str_labels = create_windows(df)           # (N, T, C), (N,)
    features = extract_features(windows)               # (N, FEATURE_VECTOR_SIZE)

    label_to_idx = {lbl: i for i, lbl in enumerate(ASL_LABELS)}
    int_labels = np.array(
        [label_to_idx.get(lbl, 0) for lbl in str_labels], dtype=np.int64
    )

    rng = np.random.default_rng(seed)
    idx = np.arange(len(features))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    def _make_ds(mask: np.ndarray) -> TensorDataset:
        X = torch.from_numpy(features[mask].astype(np.float32))
        y = torch.from_numpy(int_labels[mask])
        return TensorDataset(X, y)

    return _make_ds(train_idx), _make_ds(val_idx)


def _load_cross_modal_dataset(
    data_dir: str,
    video_dir: str,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Load paired (EMG flat window, landmark vector) data for cross-modal training.

    Loads EMG windows from *data_dir* and hand landmark .npy files from
    *video_dir* (produced by scripts/extract_wlasl_landmarks.py or by the
    HandLandmarkExtractor pipeline).

    Falls back to synthetic landmark generation when video_dir landmarks are
    unavailable for a class (same strategy as train_cross_modal() in
    cross_modal_embedding.py).

    Parameters
    ----------
    data_dir:
        Directory containing labeled EMG session CSV files.
    video_dir:
        Directory containing ``landmarks.npz`` produced by
        scripts/extract_wlasl_landmarks.py.
    seed:
        Random seed for the train/val split.

    Returns
    -------
    (train_dataset, val_dataset)
        TensorDatasets containing (emg_flat, landmark, label_index) triplets.
        emg_flat shape: (N, WINDOW_SIZE_SAMPLES * N_CHANNELS)
        landmark shape: (N, 63)
        label_index shape: (N,) int64
    """
    from src.data.loader import create_windows, load_dataset

    # EMG windows.
    df = load_dataset(data_dir)
    windows, str_labels = create_windows(df)           # (N, T, C), (N,)
    N, T, C = windows.shape
    emg_flat = windows.reshape(N, T * C).astype(np.float32)   # (N, 320)

    # Vision landmarks.
    landmarks_path = Path(video_dir) / "landmarks.npz"
    if landmarks_path.exists():
        archive = np.load(str(landmarks_path), allow_pickle=True)
        word_to_landmark: dict[str, np.ndarray] = dict(archive)
        rng = np.random.default_rng(seed)
        landmarks = np.empty((N, 63), dtype=np.float32)
        for i, lbl in enumerate(str_labels):
            if lbl in word_to_landmark:
                proto = word_to_landmark[lbl].astype(np.float32)
                noise = rng.standard_normal(63).astype(np.float32) * 0.02
                landmarks[i] = proto + noise
            else:
                # Fallback: random unit vector with small noise.
                raw = rng.standard_normal(63).astype(np.float32)
                landmarks[i] = raw / (np.linalg.norm(raw) + 1e-9)
        logger.info("Loaded real WLASL landmarks from %s", landmarks_path)
    else:
        logger.warning(
            "landmarks.npz not found in %s -- using synthetic landmarks. "
            "Run scripts/extract_wlasl_landmarks.py first for real cross-modal training.",
            video_dir,
        )
        rng = np.random.default_rng(seed)
        label_to_proto: dict[str, np.ndarray] = {}
        for lbl in np.unique(str_labels):
            raw = rng.standard_normal(63).astype(np.float32)
            label_to_proto[lbl] = raw / (np.linalg.norm(raw) + 1e-9)
        landmarks = np.empty((N, 63), dtype=np.float32)
        for i, lbl in enumerate(str_labels):
            noise = rng.standard_normal(63).astype(np.float32) * 0.05
            noisy = label_to_proto[lbl] + noise
            landmarks[i] = noisy / (np.linalg.norm(noisy) + 1e-9)

    label_to_idx = {lbl: i for i, lbl in enumerate(ASL_LABELS)}
    int_labels = np.array(
        [label_to_idx.get(lbl, 0) for lbl in str_labels], dtype=np.int64
    )

    rng2 = np.random.default_rng(seed + 1)
    idx = np.arange(N)
    rng2.shuffle(idx)
    split = int(N * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    def _make_ds(mask: np.ndarray) -> TensorDataset:
        emg_t = torch.from_numpy(emg_flat[mask])
        lm_t = torch.from_numpy(landmarks[mask])
        y_t = torch.from_numpy(int_labels[mask])
        return TensorDataset(emg_t, lm_t, y_t)

    return _make_ds(train_idx), _make_ds(val_idx)


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 accuracy as a fraction in [0, 1]."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def _train_lstm(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Train the LSTM classifier with DDP.

    Only rank-0 writes checkpoints and W&B logs.

    Parameters
    ----------
    args:
        Parsed command-line arguments.
    rank:
        Global rank of this process.
    world_size:
        Total number of processes.
    device:
        CUDA/CPU device for this rank.
    """
    from src.models.lstm_classifier import ASLEMGClassifier

    is_main = rank == 0

    if is_main:
        logger.info("Loading LSTM dataset from %s", args.data_dir)

    train_ds, val_ds = _load_lstm_dataset(args.data_dir)

    # Distributed samplers ensure each GPU sees a non-overlapping shard.
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = ASLEMGClassifier(
        input_size=FEATURE_VECTOR_SIZE,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        label_names=list(ASL_LABELS),
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Weights & Biases setup.
    wandb_run = None
    if args.wandb and is_main:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"lstm-ddp-{world_size}gpu",
                config={
                    "model": "lstm",
                    "epochs": args.epochs,
                    "batch_size": args.batch_size * world_size,
                    "lr": args.lr,
                    "amp": args.amp,
                    "grad_accum": args.grad_accum,
                    "world_size": world_size,
                },
            )
        except ImportError:
            logger.warning("wandb not installed -- skipping W&B logging.")

    best_val_acc = 0.0
    best_ckpt_path = output_dir / "lstm_best.pt"

    if is_main:
        logger.info(
            "Starting LSTM training: %d epochs, batch=%d, lr=%.4f, AMP=%s, "
            "grad_accum=%d, world_size=%d",
            args.epochs, args.batch_size, args.lr, args.amp, args.grad_accum, world_size,
        )

    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, (features, targets) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(features)
                loss = criterion(logits, targets) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)

        # Validation (run on all ranks; only rank-0 logs and saves).
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(features)
                val_correct += (logits.argmax(dim=-1) == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / max(val_total, 1)

        if is_main:
            logger.info(
                "Epoch %4d/%d  loss=%.4f  val_acc=%.4f  lr=%.6f",
                epoch + 1, args.epochs, avg_loss, val_acc,
                scheduler.get_last_lr()[0],
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "val/accuracy": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Unwrap DDP to get the underlying module for saving.
                raw_model = model.module if isinstance(model, DDP) else model
                raw_model.save(str(best_ckpt_path))
                logger.info(
                    "  New best val_acc=%.4f -- checkpoint saved to %s",
                    val_acc, best_ckpt_path,
                )

    if is_main:
        logger.info(
            "Training complete. Best val_acc=%.4f, checkpoint at %s",
            best_val_acc, best_ckpt_path,
        )
        if wandb_run is not None:
            wandb_run.finish()


def _train_cross_modal(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Train the CrossModalASL dual-encoder with DDP.

    Uses the InfoNCE contrastive loss over paired (EMG window, hand landmark)
    batches.  Only rank-0 saves checkpoints and W&B logs.

    Parameters
    ----------
    args:
        Parsed command-line arguments.
    rank:
        Global rank of this process.
    world_size:
        Total number of processes.
    device:
        CUDA/CPU device for this rank.
    """
    from src.models.cross_modal_embedding import CrossModalASL, EMBED_DIM, EMG_INPUT_DIM, VISION_INPUT_DIM
    import torch.nn.functional as F

    is_main = rank == 0
    video_dir = args.video_dir or ""

    if is_main:
        logger.info(
            "Loading cross-modal dataset: EMG from %s, landmarks from %s",
            args.data_dir, video_dir or "(synthetic)",
        )

    train_ds, val_ds = _load_cross_modal_dataset(
        args.data_dir, video_dir or args.data_dir
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = CrossModalASL(
        embed_dim=EMBED_DIM,
        temperature=0.07,
        emg_input_dim=EMG_INPUT_DIM,
        vision_input_dim=VISION_INPUT_DIM,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb and is_main:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"cross_modal-ddp-{world_size}gpu",
                config={
                    "model": "cross_modal",
                    "epochs": args.epochs,
                    "batch_size": args.batch_size * world_size,
                    "lr": args.lr,
                    "amp": args.amp,
                    "grad_accum": args.grad_accum,
                    "world_size": world_size,
                    "video_dir": video_dir or "synthetic",
                },
            )
        except ImportError:
            logger.warning("wandb not installed -- skipping W&B logging.")

    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "cross_modal_best.pt"

    if is_main:
        logger.info(
            "Starting CrossModalASL training: %d epochs, batch=%d, lr=%.4f, AMP=%s, "
            "grad_accum=%d, world_size=%d",
            args.epochs, args.batch_size, args.lr, args.amp, args.grad_accum, world_size,
        )

    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, (emg_batch, lm_batch, _labels) in enumerate(train_loader):
            emg_batch = emg_batch.to(device, non_blocking=True)
            lm_batch = lm_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                raw_model = model.module if isinstance(model, DDP) else model
                emg_emb, vis_emb = raw_model(emg_batch, lm_batch)
                loss = raw_model.info_nce_loss(emg_emb, vis_emb) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)

        # Validation loss.
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for emg_batch, lm_batch, _labels in val_loader:
                emg_batch = emg_batch.to(device, non_blocking=True)
                lm_batch = lm_batch.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    raw_model = model.module if isinstance(model, DDP) else model
                    emg_emb, vis_emb = raw_model(emg_batch, lm_batch)
                    v_loss = raw_model.info_nce_loss(emg_emb, vis_emb)
                val_loss_total += v_loss.item()
                val_steps += 1
        val_loss = val_loss_total / max(val_steps, 1)

        if is_main:
            logger.info(
                "Epoch %4d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.6f",
                epoch + 1, args.epochs, avg_loss, val_loss,
                scheduler.get_last_lr()[0],
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "val/loss": val_loss,
                    "lr": scheduler.get_last_lr()[0],
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                raw_model = model.module if isinstance(model, DDP) else model
                raw_model.save(str(best_ckpt_path))
                logger.info(
                    "  New best val_loss=%.4f -- checkpoint saved to %s",
                    val_loss, best_ckpt_path,
                )

    if is_main:
        logger.info(
            "Training complete. Best val_loss=%.4f, checkpoint at %s",
            best_val_loss, best_ckpt_path,
        )
        if wandb_run is not None:
            wandb_run.finish()


# ---------------------------------------------------------------------------
# CNN-LSTM dispatcher
# ---------------------------------------------------------------------------


def _train_cnn_lstm(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Train the CNN-LSTM classifier with DDP.

    Delegates to the same data pipeline as _train_lstm; the CNN-LSTM model
    expects the same FEATURE_VECTOR_SIZE input.

    Parameters
    ----------
    args:
        Parsed command-line arguments.
    rank:
        Global rank of this process.
    world_size:
        Total number of processes.
    device:
        CUDA/CPU device for this rank.
    """
    from src.models.cnn_lstm_classifier import CNNLSTMClassifier

    is_main = rank == 0

    if is_main:
        logger.info("Loading CNN-LSTM dataset from %s", args.data_dir)

    train_ds, val_ds = _load_lstm_dataset(args.data_dir)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = CNNLSTMClassifier(
        input_size=FEATURE_VECTOR_SIZE,
        num_classes=NUM_CLASSES,
        label_names=list(ASL_LABELS),
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb and is_main:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"cnn_lstm-ddp-{world_size}gpu",
                config={
                    "model": "cnn_lstm",
                    "epochs": args.epochs,
                    "batch_size": args.batch_size * world_size,
                    "lr": args.lr,
                    "amp": args.amp,
                    "grad_accum": args.grad_accum,
                    "world_size": world_size,
                },
            )
        except ImportError:
            logger.warning("wandb not installed -- skipping W&B logging.")

    best_val_acc = 0.0
    best_ckpt_path = output_dir / "cnn_lstm_best.pt"

    if is_main:
        logger.info(
            "Starting CNN-LSTM training: %d epochs, batch=%d, lr=%.4f, AMP=%s, "
            "grad_accum=%d, world_size=%d",
            args.epochs, args.batch_size, args.lr, args.amp, args.grad_accum, world_size,
        )

    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, (features, targets) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(features)
                loss = criterion(logits, targets) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(features)
                val_correct += (logits.argmax(dim=-1) == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / max(val_total, 1)

        if is_main:
            logger.info(
                "Epoch %4d/%d  loss=%.4f  val_acc=%.4f  lr=%.6f",
                epoch + 1, args.epochs, avg_loss, val_acc,
                scheduler.get_last_lr()[0],
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "val/accuracy": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                raw_model = model.module if isinstance(model, DDP) else model
                raw_model.save(str(best_ckpt_path))
                logger.info(
                    "  New best val_acc=%.4f -- checkpoint saved to %s",
                    val_acc, best_ckpt_path,
                )

    if is_main:
        logger.info(
            "Training complete. Best val_acc=%.4f, checkpoint at %s",
            best_val_acc, best_ckpt_path,
        )
        if wandb_run is not None:
            wandb_run.finish()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU EMG-ASL training via DistributedDataParallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=["lstm", "cnn_lstm", "cross_modal"],
        default="lstm",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/",
        metavar="PATH",
        help="Directory containing labeled EMG session CSV files.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory containing WLASL landmark data (landmarks.npz). "
            "Required for meaningful cross_modal training; uses synthetic "
            "landmarks when omitted."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Per-GPU mini-batch size. Effective batch = batch-size * world_size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        metavar="STEPS",
        help="Gradient accumulation steps. Effective batch is multiplied by this value.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable automatic mixed precision (bfloat16/float16) via torch.cuda.amp.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/gpu/",
        metavar="PATH",
        help="Directory where best model checkpoints are saved.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        default="maia-emg-asl",
        help="W&B project name (used when --wandb is set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting and weight initialization.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Initialize DDP, dispatch to the appropriate training function, then clean up."""
    args = parse_args()

    rank, world_size, device = setup_ddp()

    t0 = time.time()

    try:
        if args.model == "lstm":
            _train_lstm(args, rank, world_size, device)
        elif args.model == "cnn_lstm":
            _train_cnn_lstm(args, rank, world_size, device)
        elif args.model == "cross_modal":
            _train_cross_modal(args, rank, world_size, device)
        else:
            raise ValueError(f"Unknown model: {args.model}")
    finally:
        cleanup_ddp()

    if rank == 0:
        elapsed = time.time() - t0
        logger.info("Total wall time: %.1f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
