#!/bin/bash
#SBATCH --job-name=r6bevalS
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report VI, Thread B -- SIMILARITY read-out variant (EVAL-ONLY). Same as r6_b_eval.sbatch
# but on the SIMILARITY checkpoints, writing to asym_chains_sim/. Asks whether the unbalanced
# two-chain split puzzle (4+36 vs 17+23) changes under the similarity head -- which doubles the
# reach to 2*3^L=18, so the balanced 20+20 (max within-distance 19) now sits right at the wall.
# eval_asym_chains.py auto-detects the read-out, so only the checkpoint paths + output dir change.
#
# Distributions: path_union (clean, from r6_a1_train_sim), ER (n20/n40 reuse Report-IV similarity,
# n64 from r6_a1_train_sim), mixed (Report-IV similarity baseline). NO retraining here.
#   sbatch scripts/r6_b_eval_sim.sbatch
# Gate on the similarity Thread-A trainings if path_union-sim / er-n64-sim are not in yet
# (afterany, not afterok -- err. 44):
#   sbatch --dependency=afterany:<trainSimJobIdsColonSeparated> scripts/r6_b_eval_sim.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUTROOT="runs/report6/asym_chains_sim"
NG=300

find_ckpt () { for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done; echo ""; }

run () {   # run <tag> <candidate ckpt> [more candidates...]
  TAG="$1"; shift
  CKPT="$(find_ckpt "$@")"
  if [ -z "$CKPT" ]; then echo "-- skip $TAG (no checkpoint among: $*)"; return; fi
  echo "=== $TAG  <-  $CKPT ==="
  python eval_asym_chains.py --checkpoint "$CKPT" --output_dir "$OUTROOT/$TAG" --n_graphs "$NG"
}

# ---- path_union (clean, similarity), n20 / n40 / n64 -- 4 seeds (Thread A1-sim) ----
for N in 20 40 64; do for S in 1000 2000 3000 4000; do
  run "n${N}_pathunion_seed${S}" "runs/report6/a1_train/n${N}_path_union_roberta_similarity_lam0_seed${S}/last.pt"
done; done

# ---- ER (similarity, unseen cross-check) ----
# n20: Report-IV similarity ER (8 seeds)
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  run "n20_er_seed${S}" \
    "runs/report4/families_n20/n20_er_roberta_similarity_lam0_seed${S}/last.pt" \
    "runs/families_n20/n20_er_roberta_similarity_lam0_seed${S}/last.pt"
done
# n40: Report-IV similarity ER (4 seeds)
for S in 1000 2000 3000 4000; do
  run "n40_er_seed${S}" \
    "runs/report4/families_n40/n40_er_roberta_similarity_lam0_seed${S}/last.pt" \
    "runs/families_n40/n40_er_roberta_similarity_lam0_seed${S}/last.pt"
done
# n64: similarity, trained for Report VI (Thread A1-sim)
for S in 1000 2000 3000 4000; do
  run "n64_er_seed${S}" "runs/report6/a1_train/n64_er_roberta_similarity_lam0_seed${S}/last.pt"
done

# ---- mixed (Report-IV similarity baseline), n20 / n40 -- 4 seeds ----
for N in 20 40; do for S in 1000 2000 3000 4000; do
  run "n${N}_mixed_seed${S}" \
    "runs/report4/families_n${N}/n${N}_mixed_roberta_similarity_lam0_seed${S}/last.pt" \
    "runs/families_n${N}/n${N}_mixed_roberta_similarity_lam0_seed${S}/last.pt"
done; done

echo "ALL DONE. jsons -> $OUTROOT/<tag>/asym_chains.json   (count them; eval skips missing ckpts)"
