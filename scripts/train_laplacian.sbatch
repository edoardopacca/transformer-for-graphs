#!/bin/bash
#SBATCH --job-name="laplacian"
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
#SBATCH --array=0-3

mkdir -p out err
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf

# Experiment B: does a spectral/Laplacian bias move the 3^L=9 wall? (L=2, n=64).
# array task -> (readout, lambda_lap):
#   0: linear     lam=0   (baseline = depth-sweep L=2, matrix-powering)
#   1: similarity lam=0   (similarity read-out only)
#   2: similarity lam=1   (similarity read-out + Laplacian smoothness)
#   3: linear     lam=1   (Laplacian smoothness only, standard read-out)
case $SLURM_ARRAY_TASK_ID in
  0) RO=linear;     LAM=0 ;;
  1) RO=similarity; LAM=0 ;;
  2) RO=similarity; LAM=1 ;;
  3) RO=linear;     LAM=1 ;;
esac

echo "Node: $(hostname) | readout=$RO lambda=$LAM"
python experiments2/train_laplacian.py \
    --output_root "runs/laplacian" \
    --n_nodes 64 --n_layers 2 \
    --readout $RO --lambda_lap $LAM \
    --num_workers 16 --train_steps 500000 --batch_size 1000 --seed 1000
