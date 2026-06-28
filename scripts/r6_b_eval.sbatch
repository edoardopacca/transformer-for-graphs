#!/bin/bash
#SBATCH --job-name=r6beval
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report VI, Thread B -- asymmetric two-chains (EVAL-ONLY). Runs eval_asym_chains.py on
# the existing base linear checkpoints, sweeping the split (a, n-a): whole-graph exact
# match + the within-long / within-short / cut decomposition + the long-component
# per-distance reach profile, all BY SPLIT. Answers the Report-IV puzzle (is 4+36 easier
# than 17+23?) as the full curve over splits.
#
# Distributions (Report VI data principle, one named training set per checkpoint):
#   - path_union : the clean "reasoning along a path" condition (sees single-path reach
#                  and random 2-way splits) -- the clean read. From Thread A1 (n20/40/64).
#   - ER         : never saw any path structure (prof's "unseen" cross-check). n20/n40
#                  REUSE existing checkpoints; n64 from Thread A1.
#   - mixed      : the Report-IV reference (NOT a clean condition; eval'd as a baseline).
# NO retraining: all checkpoints already exist, so this slots in whenever a GPU is free.
# Eval-only -> short partition. ~36 checkpoints; if it ever nears the 1h10 cap, comment
# out the n64 block (or one condition) and resubmit -- it is re-runnable and additive.
#
#   sbatch scripts/r6_b_eval.sbatch
# Optionally gate on the Thread-A trainings (afterany, not afterok -- err. 44):
#   sbatch --dependency=afterany:<trainJobIdsColonSeparated> scripts/r6_b_eval.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUTROOT="runs/report6/asym_chains"
NG=300

# find the first existing checkpoint among the candidates (bucketed vs non-bucketed .pt
# path ambiguity on HPC; see istruzioni err. 34)
find_ckpt () {
  for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo ""
}

run () {   # run <tag> <candidate ckpt> [more candidates...]
  TAG="$1"; shift
  CKPT="$(find_ckpt "$@")"
  if [ -z "$CKPT" ]; then echo "-- skip $TAG (no checkpoint among: $*)"; return; fi
  echo "=== $TAG  <-  $CKPT ==="
  python eval_asym_chains.py --checkpoint "$CKPT" --output_dir "$OUTROOT/$TAG" --n_graphs "$NG"
}

# ---- path_union (clean), n20 / n40 / n64 -- 4 seeds (Thread A1) ----
for N in 20 40 64; do for S in 1000 2000 3000 4000; do
  run "n${N}_pathunion_seed${S}" "runs/report6/a1_train/n${N}_path_union_roberta_linear_lam0_seed${S}/last.pt"
done; done

# ---- ER (unseen cross-check) ----
# n20: Report-III roberta unrestricted = ER/linear baseline (8 seeds)
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  run "n20_er_seed${S}" \
    "runs/report3/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt" \
    "runs/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt"
done
# n40: Report-IV families_n40 ER linear (4 seeds)
for S in 1000 2000 3000 4000; do
  run "n40_er_seed${S}" \
    "runs/report4/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt" \
    "runs/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt"
done
# n64: trained for Report VI (Thread A1)
for S in 1000 2000 3000 4000; do
  run "n64_er_seed${S}" "runs/report6/a1_train/n64_er_roberta_linear_lam0_seed${S}/last.pt"
done

# ---- mixed (Report-IV baseline), n20 / n40 -- 4 seeds ----
for N in 20 40; do for S in 1000 2000 3000 4000; do
  run "n${N}_mixed_seed${S}" \
    "runs/report4/families_n${N}/n${N}_mixed_roberta_linear_lam0_seed${S}/last.pt" \
    "runs/families_n${N}/n${N}_mixed_roberta_linear_lam0_seed${S}/last.pt"
done; done

echo "ALL DONE. jsons -> $OUTROOT/<tag>/asym_chains.json   (count them; eval skips missing ckpts)"
