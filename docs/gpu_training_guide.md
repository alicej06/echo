# GPU Training Guide — IYA Nvidia Lab

This guide covers how to run EMG-ASL model training on the **IYA Nvidia Lab
cluster** — the same 18-GPU cluster used for the Nalana project.

**Cluster specs:**
- **18x NVIDIA RTX A6000** (48 GB VRAM each) — total 864 GB VRAM
- CUDA 12.1+, PyTorch 2.x
- Workspace at `/workspace`
- Conda-based Python environments (Python 3.11)

The A6000's 48 GB VRAM is far more than needed for the EMG-ASL LSTM
(~1 M parameters). A single A6000 can train on the entire GRABMyo + Italian SL
+ NinaProDB combined dataset in under 10 minutes. Multi-GPU is used for
hyperparameter sweeps and cross-modal embedding training.

---

## 1. SSH and Initial Orientation

```bash
ssh user@gpu-cluster-host

# Verify GPU inventory
nvidia-smi --list-gpus
# Expect: 18 GPUs listed (RTX A6000)

# Check CUDA version
nvcc --version
# Need: CUDA 12.1+

# Check disk space (GRABMyo alone is ~9.4 GB)
df -h /workspace
# Need: ≥ 50 GB free for datasets + models
```

---

## 2. Clone and Environment Setup

```bash
cd /workspace

git clone https://github.com/calebnewtonusc/emg-asl-layer.git
cd emg-asl-layer

# Create conda environment (Python 3.11 — matches Nalana cluster setup)
conda create -n emg-asl python=3.11 -y
conda activate emg-asl

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify PyTorch sees GPUs
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
# Expect: 18 GPUs

# Optional: Weights & Biases for training metrics
pip install wandb
wandb login   # paste API key from wandb.ai/authorize
```

---

## 3. Copy Data to the Cluster

**Option A — rsync from local machine:**

```bash
rsync -avz --progress \
    data/raw/ \
    user@gpu-cluster-host:/workspace/emg-asl-layer/data/raw/
```

**Option B — download public datasets directly on the cluster (recommended):**

```bash
# Italian Sign Language (~50 MB, instant — best starting dataset)
bash scripts/download_datasets.sh --italian-sl
python scripts/prepare_italian_sl.py --data-dir data/external/italian-sl/

# GRABMyo (~9.4 GB — run in background)
nohup wget -r -N -c -np \
    https://physionet.org/files/grabmyo/1.1.0/ \
    -P data/external/grabmyo/ \
    > logs/grabmyo_download.log 2>&1 &
echo "GRABMyo downloading in background — check logs/grabmyo_download.log"
```

---

## 4. Quick Training (Single GPU, < 5 min)

For rapid iteration on a single A6000:

```bash
# Train LSTM on Italian SL data (~3 min on 1x A6000)
CUDA_VISIBLE_DEVICES=0 python scripts/train_real.py \
    --data-dir data/raw/italian_sl/ \
    --model lstm \
    --epochs 50 \
    --batch-size 2048 \
    --amp \
    --output-dir models/

# Verify the output
ls models/
# asl_emg_classifier.pt
# asl_emg_classifier.onnx
```

---

## 5. Full Multi-GPU Training

### Workflow A: LSTM — 4 GPUs, ~10 min, all datasets

```bash
torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \
    --model lstm \
    --data-dir data/raw/ \
    --epochs 200 \
    --batch-size 8192 \
    --amp \
    --wandb \
    --wandb-project maia-emg-asl \
    --output-dir models/gpu/
```

### Workflow B: CNN-LSTM — 4 GPUs, ~15 min

```bash
torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \
    --model cnn_lstm \
    --data-dir data/raw/ \
    --epochs 200 \
    --batch-size 4096 \
    --amp \
    --output-dir models/gpu/
```

### Workflow C: Cross-Modal Embedding — 8 GPUs, ~2–4 hours

Aligns EMG embeddings with MediaPipe hand landmark prototypes from sign
language videos, enabling zero-shot vocabulary extension beyond 36 classes.

```bash
# Step 1: Extract hand landmarks from sign video dataset (CPU, ~1h)
python scripts/extract_wlasl_landmarks.py \
    --video-dir data/wlasl/videos/ \
    --output-dir data/wlasl/

# Step 2: Train cross-modal embeddings
torchrun --nproc_per_node=8 scripts/train_gpu_ddp.py \
    --model cross_modal \
    --emg-dir data/raw/ \
    --landmark-dir data/wlasl/ \
    --epochs 100 \
    --batch-size 2048 \
    --amp \
    --output-dir models/gpu/
```

### Workflow D: Optuna HPO Sweep — 2 GPUs, ~1–2 hours

Run before committing to a long training run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/optuna_hpo.py \
    --model lstm \
    --data-dir data/raw/ \
    --n-trials 50 \
    --output-dir models/hpo/

# Best hyperparameters:
cat models/hpo/best_hparams.json
```

---

## 6. Recommended Batch Sizes (RTX A6000, 48 GB)

The EMG-ASL LSTM is tiny — A6000 VRAM is not the bottleneck. Scale batch
size to saturate GPU compute throughput:

| GPUs | `--batch-size` | Effective batch | Notes |
|------|---------------|-----------------|-------|
| 1    | 2048          | 2048            | Rapid single-GPU iteration |
| 4    | 8192          | 32768           | Standard multi-GPU run |
| 8    | 8192          | 65536           | Cross-modal embedding training |
| 18   | 4096          | 73728           | Full cluster HPO sweeps |

Scale learning rate linearly with effective batch:
`lr = 1e-3 × (effective_batch / 128)`

Example: 4 GPUs, batch 8192 → `lr ≈ 6.4e-2` (use cosine decay with warmup).

---

## 7. Monitor Training

```bash
# GPU utilization (run in a second terminal)
watch -n 5 nvidia-smi

# W&B dashboard — visit:
# https://wandb.ai/[your-username]/maia-emg-asl

# Training log
tail -f logs/train.log
```

---

## 8. After Training

```bash
# Best checkpoint is at:
ls models/gpu/lstm_best.pt

# Evaluate on held-out test set
python scripts/evaluate.py \
    --model models/gpu/lstm_best.pt \
    --data-dir data/raw/

# Export to ONNX for production inference
python scripts/export_onnx.py \
    --checkpoint models/gpu/lstm_best.pt \
    --output models/asl_emg_classifier.onnx

# Copy model back to local machine
rsync -avz user@gpu-cluster-host:/workspace/emg-asl-layer/models/ models/

# (Optional) Upload to HuggingFace Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='models/gpu',
    repo_id='maia-biotech/emg-asl-classifier',
    repo_type='model'
)
"

# Start local inference server with new model
./start-server.sh
```

---

## 9. W&B Metrics Logged

| Metric | Description |
|--------|-------------|
| `train/loss` | Cross-entropy loss per epoch |
| `val/accuracy` | Top-1 validation accuracy |
| `val/loss` | Validation cross-entropy |
| `lr` | Current learning rate (cosine schedule) |
| `epoch` | Training epoch |

Only the rank-0 process (GPU 0) writes to W&B — no duplicate entries in
multi-GPU runs (same pattern as Nalana training).

---

## 10. Troubleshooting

### "CUDA out of memory"

The EMG LSTM should never OOM on A6000 with batch sizes below 32768.
If it happens during cross-modal training:

```bash
torchrun --nproc_per_node=8 scripts/train_gpu_ddp.py \
    --batch-size 1024 \
    --grad-accum 2 \
    --amp
```

### "NCCL error: unhandled system error"

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1   # disable InfiniBand if unavailable
```

### Wrong GPUs / conflict with other jobs

```bash
# See what's running on each GPU
nvidia-smi

# Explicitly assign specific GPUs
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ...
```

### `mediapipe` not found (landmark extraction)

```bash
pip install 'mediapipe>=0.10' 'opencv-python>=4.9'
```

### `myo-python` on cluster (not needed for training)

The MYO armband is not used on the cluster — training uses pre-recorded CSV
files. myo-python is a laptop-only dependency for live data collection.
The `requirements.txt` marks it as optional; it won't be installed unless
the MYO SDK is present.
