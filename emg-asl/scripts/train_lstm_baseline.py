#!/usr/bin/env python3
"""
Train LSTM baseline on synthetic data, export to ONNX.

Usage:
    python scripts/train_lstm_baseline.py
    python scripts/train_lstm_baseline.py --epochs 100
    python scripts/train_lstm_baseline.py --export-onnx
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import (
    N_CHANNELS, WINDOW_SAMPLES, N_CLASSES, ASL_CLASSES,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
)
from src.data.loader import generate_synthetic_dataset
from src.models.lstm_classifier import LSTMClassifier


def train(args):
    # ---- Data ----
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_dataset(n_samples_per_class=200, seed=42)
    n = len(y)
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Classes: {N_CLASSES}")

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)

    # ---- Model ----
    model = LSTMClassifier(
        n_channels=N_CHANNELS, n_classes=N_CLASSES,
        hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    epochs = args.epochs

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Acc':>10} {'LR':>12}")
    print("-" * 46)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
        scheduler.step()

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb.to(device)).argmax(1)
                    correct += (preds == yb.to(device)).sum().item()
            val_acc = correct / len(y_val)
            lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>6} {total_loss/len(y_tr):>12.4f} {val_acc:>10.3f} {lr:>12.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stop at epoch {epoch}")
                    break

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    # Save model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "asl_emg_classifier.pt"
    torch.save(best_state, pt_path)
    print(f"Saved: {pt_path}")

    # Export ONNX
    if args.export_onnx or True:  # always export
        model.load_state_dict(best_state)
        model.eval()
        dummy = torch.zeros(1, WINDOW_SAMPLES, N_CHANNELS)
        onnx_path = out_dir / "asl_emg_classifier.onnx"
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1e3:.1f} KB)")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train LSTM baseline on synthetic EMG data")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX (always done by default)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
