#!/bin/bash
#SBATCH --job-name=bridged
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report V, core experiment: bridged-cliques vs split-cliques on the BASE model
# (RoBERTa, linear read-out). Eval-only -- runs every connectivity checkpoint that
# EXISTS under runs/report5; missing ones are skipped. (The older Report-III/IV base
# checkpoints were cleaned for quota, so we use: n20 ER from the density sweep p08
# runs, and n20 mixed / n40 from the base retrains of scripts/train_base_bridged_*.)
# ~1-3 min per checkpoint. The train_base_bridged_* scripts also auto-run this eval
# on their own freshly trained checkpoint, so this file is mostly for the reused
# n20-ER checkpoints + re-running after a code change.
#
#   sbatch scripts/eval_bridged_cliques.sbatch
# Then locally:  git pull ; python plot_bridged_cliques.py

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/report5/bridged_cliques"
NG=2000

run_one () {   # $1 = checkpoint, $2 = tag (n{N}_{set}_seed{S})
  if [ -f "$1" ]; then
    echo "== $2  ($1) =="
    python eval_bridged_cliques.py --checkpoint "$1" --output_dir "$OUT_ROOT/$2" --n_graphs "$NG"
  else
    echo "-- skip $2 (no checkpoint at $1)"
  fi
}

# n20 ER linear: REUSE the density-sweep p08 runs (= ER(20,0.08) linear, 4 seeds)
for S in 1000 2000 3000 4000; do
  run_one "runs/report5/density_sweep/p08/n20_er_roberta_linear_lam0_seed${S}/last.pt" "n20_er_seed${S}"
done
# n20 mixed linear + n40 er/mixed linear: from the base retrains (if present)
for S in 1000 2000 3000 4000; do
  run_one "runs/report5/base_n20/n20_mixed_roberta_linear_lam0_seed${S}/last.pt" "n20_mixed_seed${S}"
  run_one "runs/report5/base_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt"    "n40_er_seed${S}"
  run_one "runs/report5/base_n40/n40_mixed_roberta_linear_lam0_seed${S}/last.pt" "n40_mixed_seed${S}"
done

echo "bridged-cliques eval done -> $OUT_ROOT"
