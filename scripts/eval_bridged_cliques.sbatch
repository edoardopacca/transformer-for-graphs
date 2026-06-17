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
# (RoBERTa, linear read-out). Eval-only on existing checkpoints -- no training.
# Runs every base linear checkpoint we have (ER + mixed, n=20 + n=40, all seeds);
# missing checkpoints are skipped. ~1-3 min per checkpoint, so one job covers all.
#
#   sbatch scripts/eval_bridged_cliques.sbatch
#
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

# ER, n=20, linear  (Report-III RoBERTa baselines, best.pt; 8 seeds)
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  run_one "runs/report3/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt" "n20_er_seed${S}"
done
# mixed, n=20, linear  (Report-IV families, last.pt; 8 seeds)
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  run_one "runs/report4/families_n20/n20_mixed_roberta_linear_lam0_seed${S}/last.pt" "n20_mixed_seed${S}"
done
# ER + mixed, n=40, linear  (Report-IV families_n40, last.pt; 4 seeds)
for S in 1000 2000 3000 4000; do
  run_one "runs/report4/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt"    "n40_er_seed${S}"
  run_one "runs/report4/families_n40/n40_mixed_roberta_linear_lam0_seed${S}/last.pt" "n40_mixed_seed${S}"
done

echo "bridged-cliques eval done -> $OUT_ROOT"
