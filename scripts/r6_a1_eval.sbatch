#!/bin/bash
#SBATCH --job-name=r6a1eval
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report VI, Thread A1 -- the multipath probe (EVAL-ONLY). Runs eval_multipath.py on the
# base linear checkpoints, sweeping (k, ell): pair (s,t) accuracy + whole-matrix metrics +
# the per-path mechanism (how many routes are intact -> single-path vs multipath reading).
#
# Distributions (Report VI data principle, one named training set per checkpoint):
#   - ER          : the "never saw any path structure" condition (prof's framing).
#                   n20/n40 REUSE existing checkpoints; n10/n64 are trained by r6_a1_train.sbatch.
#   - path_union  : sees single-path reach but NEVER parallel routes -> the clean rescue test.
#                   all sizes trained by r6_a1_train.sbatch.
#   - mixed       : the Report-IV reference (NOT a clean condition; eval'd only for comparison).
# Eval-only -> short partition. Re-runnable and order-INDEPENDENT: the reused ER/mixed
# checkpoints already exist (eval'd whenever this runs), and the freshly-trained ones are
# auto-eval'd by r6_a1_train.sbatch itself, so coverage is complete either way. For one clean
# COMPLETE final pass (single source of truth, robust to a training that died before its
# auto-eval), gate a second run on the training arrays (afterany, not afterok -- err. 44):
#   sbatch --dependency=afterany:<trainJobIdsColonSeparated> scripts/r6_a1_eval.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

ELLS="3 5 7 9 11 13 15 17 19"
KS="1 2 3 4"
OUTROOT="runs/report6/multipath"

# find the first existing checkpoint among the candidates (handles the bucketed vs
# non-bucketed .pt path ambiguity on HPC; see istruzioni err. 34)
find_ckpt () {
  for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo ""
}

run () {   # run <tag> <candidate ckpt> [more candidates...]
  TAG="$1"; shift
  CKPT="$(find_ckpt "$@")"
  if [ -z "$CKPT" ]; then echo "-- skip $TAG (no checkpoint among: $*)"; return; fi
  echo "=== $TAG  <-  $CKPT ==="
  python eval_multipath.py --checkpoint "$CKPT" --output_dir "$OUTROOT/$TAG" \
      --ks $KS --ells $ELLS --n_graphs 400
}

# ---- ER, n20 (reuse Report-III roberta unrestricted = ER/linear baseline, 8 seeds) ----
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  run "n20_er_seed${S}" \
    "runs/report3/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt" \
    "runs/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt"
done
# ---- ER, n40 (reuse Report-IV families_n40 ER linear, 4 seeds) ----
for S in 1000 2000 3000 4000; do
  run "n40_er_seed${S}" \
    "runs/report4/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt" \
    "runs/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt"
done
# ---- ER, n10 + n64 (newly trained for Report VI) ----
for N in 10 64; do for S in 1000 2000 3000 4000; do
  run "n${N}_er_seed${S}" "runs/report6/a1_train/n${N}_er_roberta_linear_lam0_seed${S}/last.pt"
done; done
# ---- path_union, all sizes (newly trained for Report VI) ----
for N in 10 20 40 64; do for S in 1000 2000 3000 4000; do
  run "n${N}_pathunion_seed${S}" "runs/report6/a1_train/n${N}_path_union_roberta_linear_lam0_seed${S}/last.pt"
done; done
# ---- mixed reference (Report IV), n20/n40, for comparison only ----
for N in 20 40; do for S in 1000 2000 3000 4000; do
  run "n${N}_mixed_seed${S}" \
    "runs/report4/families_n${N}/n${N}_mixed_roberta_linear_lam0_seed${S}/last.pt" \
    "runs/families_n${N}/n${N}_mixed_roberta_linear_lam0_seed${S}/last.pt"
done; done

echo "ALL DONE. jsons -> $OUTROOT/<tag>/multipath.json   (count them; eval skips missing ckpts)"
