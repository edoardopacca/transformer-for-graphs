#!/bin/bash
#SBATCH --job-name=diffmap
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Difficulty map (Report IV, "first part": accuracy vs diameter vs spectral gap).
# Eval-only on the STANDARD LINEAR checkpoints; no training. Loops over the
# n=20 ER-trained reproduction seeds (8) and the n=40 ER-linear family seeds (3),
# dumping per-graph (diam, gap, mean_degree, ncomp, exact, pairwise) for the 2-D
# map + partial-dependence analysis.

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

# ---- n=20 standard linear (ER-trained reproduction, 8 seeds) ----
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  CKPT="runs/report3/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt"
  if [ -f "$CKPT" ]; then
    echo "=== n20 seed $S ==="
    python eval_difficulty_map.py --checkpoint "$CKPT" \
        --output_dir "$(dirname "$CKPT")/difficulty_map" --n_nodes 20 --size_per 300
  else
    echo "MISSING $CKPT"
  fi
done

# ---- n=40 standard linear (ER-trained family runs) ----
for S in 1000 2000 3000 4000; do
  CKPT="runs/report4/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt"
  if [ -f "$CKPT" ]; then
    echo "=== n40 seed $S ==="
    python eval_difficulty_map.py --checkpoint "$CKPT" \
        --output_dir "$(dirname "$CKPT")/difficulty_map" --n_nodes 40 --size_per 200
  else
    echo "MISSING $CKPT"
  fi
done
