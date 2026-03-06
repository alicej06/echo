#!/usr/bin/env bash
# SLURM job script: train the CrossModalASL dual-encoder on the IYA Nvidia Lab cluster.
#
# This job trains the CLIP-inspired EMG + vision contrastive model against the
# WLASL landmark dataset extracted by scripts/extract_wlasl_landmarks.py.
# It requires data/wlasl/landmarks.npz to be present.
#
# Download and extract WLASL landmarks first:
#   python scripts/download_wlasl.py --extract-landmarks --output-dir data/wlasl/
#   python scripts/extract_wlasl_landmarks.py
#
# Submit:  sbatch scripts/slurm/train_cross_modal.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/slurm/<JOB_ID>_cross_modal.out

#SBATCH --job-name=maia-cross-modal
#SBATCH --partition=gpu          # adjust to IYA lab partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8      # 8 GPUs for the larger cross-modal run
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8        # 8 CPU threads per GPU for data loading
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_cross_modal.out
#SBATCH --error=logs/slurm/%j_cross_modal.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=calebnewtonusc@gmail.com

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

module load cuda/12.4
module load python/3.11

source venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/slurm

echo "=========================================="
echo "SLURM job ID   : $SLURM_JOB_ID"
echo "Node           : $(hostname)"
echo "GPUs assigned  : $CUDA_VISIBLE_DEVICES"
echo "World size     : $SLURM_NTASKS_PER_NODE"
echo "Python         : $(python --version)"
echo "PyTorch        : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count      : $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# Verify that WLASL landmarks have been extracted.
if [ ! -f data/wlasl/landmarks.npz ]; then
    echo "ERROR: data/wlasl/landmarks.npz not found."
    echo "Run the following first:"
    echo "  python scripts/download_wlasl.py --extract-landmarks --output-dir data/wlasl/"
    echo "  python scripts/extract_wlasl_landmarks.py"
    exit 1
fi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

srun torchrun \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$(hostname)" \
    --master_port=29501 \
    scripts/train_gpu_ddp.py \
        --model cross_modal \
        --data-dir data/raw/ \
        --video-dir data/wlasl/ \
        --epochs 100 \
        --batch-size 256 \
        --lr 3e-4 \
        --amp \
        --output-dir models/gpu/

# ---------------------------------------------------------------------------
# Artifact upload
# ---------------------------------------------------------------------------
# Only run on the SLURM master node (SLURM_PROCID 0).
# Skip silently if R2 credentials are not present in the environment.

if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
    if [ -n "${R2_ACCOUNT_ID:-}" ] && [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
        echo "[artifact] Uploading trained model to R2..."
        python scripts/upload_artifact.py \
            --file models/gpu/cross_modal_asl.onnx \
            --tag "slurm_${SLURM_JOB_ID}" \
            --set-latest
        echo "[artifact] Upload complete. Set R2_MODEL_URL in Railway to use this model."
    else
        echo "[artifact] R2 credentials not set. Skipping upload."
        echo "[artifact] To enable: export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=..."
    fi
fi

echo "Job finished at $(date)"
