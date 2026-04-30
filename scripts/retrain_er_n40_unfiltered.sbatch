#!/bin/bash
#SBATCH --job-name="er_n40_unfiltered"
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=08:00:00

mkdir -p out err

# ── Disable numpy thread oversubscription (we use multiprocessing, not threading)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

echo "=== Job info ==="
echo "Node:     $(hostname)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "================"

python experiments2/retrain_and_test_er_n40.py \
    --output_root "runs/retrain_er_n40_${SLURM_JOB_ID}" \
    --num_workers 16
