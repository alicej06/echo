# Training Configuration Files

This directory contains YAML configuration files for training EMG-ASL models.
Each file documents a specific model architecture and hardware target.

## Config files

| File | Model | Target hardware | Est. training time |
|---|---|---|---|
| `lstm_default.yaml` | LSTM classifier | CPU or single GPU | 5-10 min (50 epochs on Italian SL) |
| `conformer_gpu.yaml` | Conformer (attention + conv) | 4x A100 / H100 (IYA lab) | 8-12 min (300 epochs) |
| `cross_modal_wlasl.yaml` | Cross-modal LSTM + vision | 8x GPU, 128 GB RAM | ~2 hrs (100 epochs) |

## What each file is for

**`lstm_default.yaml`**
The baseline model. Trains a 2-layer LSTM on flat windowed EMG vectors
(40 samples x 8 channels = 320 features). This is the recommended starting
point for new data and the model trained by `run_full_pipeline.sh`.

**`conformer_gpu.yaml`**
A deeper conformer architecture designed for the IYA Nvidia Lab cluster.
Uses a 400 ms analysis window (2x the default) and 75% overlap to preserve
temporal resolution despite the larger window. Requires at least one modern
GPU with AMP support.

**`cross_modal_wlasl.yaml`**
Contrastive cross-modal training that aligns EMG embeddings with MediaPipe
hand-landmark embeddings extracted from the WLASL-2000 video dataset.
Run `python scripts/download_wlasl.py --extract-landmarks` before using
this config to produce `data/wlasl/landmarks.npz`.

## How to use a config with train_gpu_ddp.py

Config file loading via `--config` is on the roadmap. Currently, pass the
equivalent fields as command-line arguments directly to `train_gpu_ddp.py`.
Each YAML key maps to a CLI flag as shown in the comment block at the top
of every config file.

Single-GPU example using the values from `lstm_default.yaml`:

```bash
python scripts/train_gpu_ddp.py \
    --model lstm \
    --data-dir data/raw/ \
    --epochs 200 \
    --batch-size 256 \
    --lr 1.0e-3 \
    --amp \
    --output-dir models/gpu/ \
    --wandb \
    --wandb-project maia-emg-asl
```

Multi-GPU example using the values from `conformer_gpu.yaml`:

```bash
torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \
    --model lstm \
    --data-dir data/raw/ \
    --epochs 300 \
    --batch-size 1024 \
    --lr 3.0e-4 \
    --amp \
    --output-dir models/gpu/ \
    --wandb \
    --wandb-project maia-emg-asl
```

Cross-modal example using the values from `cross_modal_wlasl.yaml`:

```bash
torchrun --nproc_per_node=8 scripts/train_gpu_ddp.py \
    --model cross_modal \
    --data-dir data/raw/ \
    --video-dir data/wlasl/ \
    --epochs 100 \
    --batch-size 512 \
    --lr 1.0e-4 \
    --grad-accum 4 \
    --amp \
    --output-dir models/gpu/ \
    --wandb \
    --wandb-project maia-emg-asl
```

## Signal constant reference

The input dimensions in the configs derive directly from
`src/utils/constants.py`:

| Constant | Value | Role in config |
|---|---|---|
| `SAMPLE_RATE` | 200 Hz | basis for window duration |
| `WINDOW_SIZE_MS` | 200 ms | default window |
| `WINDOW_SIZE_SAMPLES` | 40 | `data.window_size_samples` |
| `N_CHANNELS` | 8 | channels per window |
| `OVERLAP` | 0.5 | `data.overlap` |
| `NUM_CLASSES` | 36 | 26 letters + 10 words |

`input_size = WINDOW_SIZE_SAMPLES * N_CHANNELS = 40 * 8 = 320`

For the cross-modal config, `vision_input_dim = 63` because MediaPipe
outputs 21 hand landmarks each with 3 coordinates (x, y, z).
