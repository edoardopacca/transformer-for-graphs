#!/bin/bash
#SBATCH --job-name=broldsd
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report V: run the bridged-cliques probe on the ORIGINAL Report-IV base checkpoints
# (linear read-out), which exist on HPC at the non-bucketed families_n* paths (they
# were NOT deleted -- see istruzioni error 34 correction). EVAL-ONLY.
# Two payoffs:
#   (1) MORE SEEDS for the clean n20-mixed result: families_n20 has seeds 1000-8000,
#       sec 5.2 only used 4. The seeds 5000-8000 are new -> pooled into the same
#       runs/report5/bridged_cliques/ dir so plot_bridged_cliques.py picks up 8 seeds.
#   (2) CONSISTENCY cross-check: seeds 1000-4000 of families_* should reproduce the
#       base_* retrains (same config+seed). Written to a SEPARATE _xcheck dir so they
#       do not overwrite the sec-5.2 results.
#
#   sbatch scripts/eval_bridged_oldseeds.sbatch
# Then locally:  git pull ; python plot_bridged_cliques.py

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

NG=2000
POOL="runs/report5/bridged_cliques"          # pooled with sec-5.2 (new seeds only)
XCHK="runs/report5/bridged_cliques_xcheck"    # consistency check (overlapping seeds)

run_one () {   # $1 = checkpoint, $2 = output dir
  if [ -f "$1" ]; then
    echo "== $2  ($1) =="
    python eval_bridged_cliques.py --checkpoint "$1" --output_dir "$2" --n_graphs "$NG"
  else
    echo "-- skip $2 (no checkpoint at $1)"
  fi
}

# (1) NEW n20-mixed seeds 5000-8000 -> pooled into sec-5.2 dir (extends 4 -> 8 seeds)
for S in 5000 6000 7000 8000; do
  run_one "runs/families_n20/n20_mixed_roberta_linear_lam0_seed${S}/last.pt" "$POOL/n20_mixed_seed${S}"
done

# (2) CONSISTENCY: overlapping seeds 1000-4000 from the ORIGINAL families checkpoints
for S in 1000 2000 3000 4000; do
  run_one "runs/families_n20/n20_mixed_roberta_linear_lam0_seed${S}/last.pt" "$XCHK/n20_mixed_seed${S}"
  run_one "runs/families_n40/n40_mixed_roberta_linear_lam0_seed${S}/last.pt" "$XCHK/n40_mixed_seed${S}"
  run_one "runs/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt"    "$XCHK/n40_er_seed${S}"
done

echo "bridged-cliques eval on old-seed base checkpoints done"
