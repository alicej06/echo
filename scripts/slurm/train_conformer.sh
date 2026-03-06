#!/usr/bin/env bash
# SLURM job script: train the Conformer classifier on the IYA Nvidia Lab cluster.
#
# NOTE: train_gpu_ddp.py must be updated to accept --model conformer before
# submitting this job.  Add the following import near the top of train_gpu_ddp.py:
#
#     from src.models.conformer_classifier import ConformerClassifier
#
# and add a branch in the model-selection block:
#
#     elif args.model == "conformer":
#         model = ConformerClassifier(
#             n_classes=NUM_CLASSES,
#             label_names=list(ASL_LABELS),
#         )
#
# The input shape (B, 40, 8) is identical to the CNN-LSTM branch so no other
# changes to the DDP training loop are required.
#
# Adjust --partition to match the actual IYA lab partition name.
# Check available partitions with: sinfo
#
# Submit:  sbatch scripts/slurm/train_conformer.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/slurm/<JOB_ID>_conformer.out

#SBATCH --job-name=maia-emg-conformer
#SBATCH --partition=gpu            # adjust to IYA lab partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # 4 GPUs per node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8          # 8 CPU threads per GPU for data loading
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%j_conformer.out
#SBATCH --error=logs/slurm/%j_conformer.err
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

# IMPORTANT: --model conformer requires train_gpu_ddp.py to import
# ConformerClassifier (see the NOTE at the top of this file).

srun torchrun \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$(hostname)" \
    --master_port=29501 \
    scripts/train_gpu_ddp.py \
        --model conformer \
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
            --file models/gpu/conformer_classifier.onnx \
            --tag "slurm_${SLURM_JOB_ID}" \
            --set-latest
        echo "[artifact] Upload complete. Set R2_MODEL_URL in Railway to use this model."
    else
        echo "[artifact] R2 credentials not set. Skipping upload."
        echo "[artifact] To enable: export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=..."
    fi
fi

echo "Job finished at $(date)"
