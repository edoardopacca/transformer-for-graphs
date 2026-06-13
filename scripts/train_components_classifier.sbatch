#!/bin/bash
#SBATCH --job-name="chaincount_n40"
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-1

mkdir -p out err

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

# 2 iterations -> seeds 1000, 2000
SEED=$(( (SLURM_ARRAY_TASK_ID + 1) * 1000 ))

echo "=== Job info ==="
echo "Node:  $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}  -> seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| bf16:', torch.cuda.is_bf16_supported())"
echo "================"

python experiments2/train_components_classifier.py \
    --output_root "runs/report3/exp3_chaincount" \
    --num_workers 16 \
    --train_steps 200000 \
    --batch_size 1000 \
    --seed ${SEED}
