#!/usr/bin/env bash
# SLURM job script: train the LSTM classifier on the IYA Nvidia Lab cluster.
#
# Adjust --partition to match the actual IYA lab partition name.
# Check available partitions with: sinfo
#
# Submit:  sbatch scripts/slurm/train_lstm.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/slurm/<JOB_ID>_lstm.out

#SBATCH --job-name=maia-emg-lstm
#SBATCH --partition=gpu          # adjust to IYA lab partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 GPUs per node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8        # 8 CPU threads per GPU for data loading
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j_lstm.out
#SBATCH --error=logs/slurm/%j_lstm.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=calebnewtonusc@gmail.com

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Load modules -- typical USC HPC module names; update if IYA lab differs.
# Run "module avail cuda" to see what is available on the cluster.
module load cuda/12.4
module load python/3.11

# Activate the project virtualenv (create it first with: python -m venv venv).
source venv/bin/activate

# Move to the project root (assumes the job was submitted from there).
# If you submit from a different directory, replace with an absolute path.
cd "$SLURM_SUBMIT_DIR"

# Create log directory in case it does not exist yet.
mkdir -p logs/slurm

# Print environment info to the output log for debugging.
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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

srun torchrun \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$(hostname)" \
    --master_port=29500 \
    scripts/train_gpu_ddp.py \
        --model lstm \
        --data-dir data/raw/ \
        --epochs 200 \
        --batch-size 512 \
        --lr 1e-3 \
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
            --file models/gpu/asl_emg_classifier.onnx \
            --tag "slurm_${SLURM_JOB_ID}" \
            --set-latest
        echo "[artifact] Upload complete. Set R2_MODEL_URL in Railway to use this model."
    else
        echo "[artifact] R2 credentials not set. Skipping upload."
        echo "[artifact] To enable: export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=..."
    fi
fi

echo "Job finished at $(date)"
