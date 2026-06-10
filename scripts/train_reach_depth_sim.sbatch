#!/bin/bash
#SBATCH --job-name="reach_sim"
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
# Same budget as the linear depth sweep (n=64, 1M steps): L=3 ~8-11h -> 14h margin.
#SBATCH --time=14:00:00
#SBATCH --array=1-3
#
# Depth sweep WITH THE SIMILARITY READ-OUT (Report IV follow-up). The linear
# sweep (train_reach_depth.sbatch) gave reach = 3^L (3, 9, 27). The n=40
# families show the similarity read-out pushes the distance wall well past 3^L on
# the SAME L=2 trunk; this asks how that wall scales with depth L:
#   - if the similarity wall also grows with L  -> the read-out buys a constant
#     multiplicative factor on an otherwise depth-bound quantity;
#   - if it is ~fixed regardless of L           -> the read-out changes the regime.
# Read it off history.json's final_per_dist (per-distance reach evolution), NOT a
# single reach radius. Runs land in runs/reach_depth/reach_n64_L{L}_..._similarity_seed1000,
# so they do not overwrite the existing linear checkpoints.

mkdir -p out err
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

# array task id == number of transformer layers (L = 1, 2, 3).
L=${SLURM_ARRAY_TASK_ID}

echo "Node: $(hostname) | L=${L} | readout=similarity (capacity 3^L matrix-powering reference)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments2/train_reach_depth.py \
    --output_root "runs/reach_depth" \
    --n_nodes 64 \
    --n_layers ${L} \
    --attn_kind normalized_relu \
    --readout similarity \
    --num_workers 16 \
    --train_steps 1000000 \
    --batch_size 1000 \
    --seed 1000
