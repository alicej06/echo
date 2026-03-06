#!/usr/bin/env bash
# SLURM job script: run Optuna hyperparameter search on the IYA Nvidia Lab cluster.
#
# Runs 50 Optuna TPE trials on 2 GPUs.  Each trial trains for up to 30 epochs
# with early stopping.  The best configuration is saved to models/hpo/best_hparams.json.
#
# After the sweep finishes the best hyperparameters are automatically used to
# train a final model for the full epoch budget (see --train-best flag in
# scripts/optuna_hpo.py).
#
# Submit:  sbatch scripts/slurm/optuna_hpo.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/slurm/<JOB_ID>_hpo.out

#SBATCH --job-name=maia-emg-hpo
#SBATCH --partition=gpu          # adjust to IYA lab partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # Optuna runs single-process; GPUs shared via CUDA
#SBATCH --gres=gpu:2             # 2 GPUs available for trials
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%j_hpo.out
#SBATCH --error=logs/slurm/%j_hpo.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=calebnewtonusc@gmail.com

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

module load cuda/12.4
module load python/3.11

source venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/slurm models/hpo

echo "=========================================="
echo "SLURM job ID   : $SLURM_JOB_ID"
echo "Node           : $(hostname)"
echo "GPUs assigned  : $CUDA_VISIBLE_DEVICES"
echo "Python         : $(python --version)"
echo "PyTorch        : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count      : $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# ---------------------------------------------------------------------------
# Optuna HPO sweep
# ---------------------------------------------------------------------------
#
# optuna_hpo.py handles CUDA device selection internally.
# CUDA_VISIBLE_DEVICES restricts it to the 2 GPUs allocated above.

python scripts/optuna_hpo.py \
    --model lstm \
    --data-dir data/raw/ \
    --n-trials 50 \
    --output-dir models/hpo/ \
    --train-best

echo "Job finished at $(date)"
