#!/bin/bash
#SBATCH --job-name="eval_reachd"
#SBATCH --account=3352759
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00

mkdir -p out err
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

echo "Node: $(hostname)"
# per-distance reach + combined figure for L=1..4 checkpoints
python eval_reach_by_distance.py --num_workers 8
