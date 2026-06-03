#!/bin/bash
#SBATCH --job-name="reach_depth"
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=1-4

mkdir -p out err
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

# Depth sweep: array task id == number of transformer layers (L = 1, 2, 3, 4).
# Prediction of the 3^L capacity theory: reach wall d* ~= 3, 9, 27, (n-limited).
L=${SLURM_ARRAY_TASK_ID}

echo "Node: $(hostname) | L=${L} (capacity 3^L)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments2/train_reach_depth.py \
    --output_root "runs/reach_depth" \
    --n_nodes 64 \
    --n_layers ${L} \
    --attn_kind normalized_relu \
    --num_workers 16 \
    --train_steps 1000000 \
    --batch_size 1000 \
    --seed 1000
