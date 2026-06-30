#!/bin/bash
#SBATCH --job-name=r6a1evalS
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report VI, Thread A1 -- SIMILARITY read-out variant of the multipath probe (EVAL-ONLY).
# Same as r6_a1_eval.sbatch but points at the SIMILARITY checkpoints and writes the json to
# the SEPARATE multipath_sim/ dir (never overwrites the linear json). eval_multipath.py
# auto-detects the read-out from the checkpoint, so the script is identical otherwise.
#
# Distributions (one named training set per checkpoint):
#   - ER          : n20/n40 REUSE the Report-IV similarity ER checkpoints; n10/n64 from r6_a1_train_sim.sbatch.
#   - path_union  : the clean rescue test; all sizes from r6_a1_train_sim.sbatch.
#   - mixed       : Report-IV similarity reference (NOT clean; eval'd for comparison).
# Eval-only -> short partition. Re-runnable and additive; skips missing checkpoints (err. 43).
#   sbatch scripts/r6_a1_eval_sim.sbatch
# For a clean final pass gated on the similarity trainings (afterany, not afterok -- err. 44):
#   sbatch --dependency=afterany:<trainSimJobIdsColonSeparated> scripts/r6_a1_eval_sim.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

ELLS="3 5 7 9 11 13 15 17 19"
KS="1 2 3 4"
OUTROOT="runs/report6/multipath_sim"

find_ckpt () { for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done; echo ""; }

run () {   # run <tag> <candidate ckpt> [more candidates...]
  TAG="$1"; shift
  CKPT="$(find_ckpt "$@")"
  if [ -z "$CKPT" ]; then echo "-- skip $TAG (no checkpoint among: $*)"; return; fi
  echo "=== $TAG  <-  $CKPT ==="
  python eval_multipath.py --checkpoint "$CKPT" --output_dir "$OUTROOT/$TAG" \
      --ks $KS --ells $ELLS --n_graphs 400
}

# ---- ER, n20 (reuse Report-IV similarity ER, 8 seeds) ----
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  run "n20_er_seed${S}" \
    "runs/report4/families_n20/n20_er_roberta_similarity_lam0_seed${S}/last.pt" \
    "runs/families_n20/n20_er_roberta_similarity_lam0_seed${S}/last.pt"
done
# ---- ER, n40 (reuse Report-IV similarity ER, 4 seeds) ----
for S in 1000 2000 3000 4000; do
  run "n40_er_seed${S}" \
    "runs/report4/families_n40/n40_er_roberta_similarity_lam0_seed${S}/last.pt" \
    "runs/families_n40/n40_er_roberta_similarity_lam0_seed${S}/last.pt"
done
# ---- ER, n10 + n64 (newly trained, similarity) ----
for N in 10 64; do for S in 1000 2000 3000 4000; do
  run "n${N}_er_seed${S}" "runs/report6/a1_train/n${N}_er_roberta_similarity_lam0_seed${S}/last.pt"
done; done
# ---- path_union, all sizes (newly trained, similarity) ----
for N in 10 20 40 64; do for S in 1000 2000 3000 4000; do
  run "n${N}_pathunion_seed${S}" "runs/report6/a1_train/n${N}_path_union_roberta_similarity_lam0_seed${S}/last.pt"
done; done
# ---- mixed similarity reference (Report IV), n20/n40, for comparison only ----
for N in 20 40; do for S in 1000 2000 3000 4000; do
  run "n${N}_mixed_seed${S}" \
    "runs/report4/families_n${N}/n${N}_mixed_roberta_similarity_lam0_seed${S}/last.pt" \
    "runs/families_n${N}/n${N}_mixed_roberta_similarity_lam0_seed${S}/last.pt"
done; done

echo "ALL DONE. jsons -> $OUTROOT/<tag>/multipath.json   (count them; eval skips missing ckpts)"
