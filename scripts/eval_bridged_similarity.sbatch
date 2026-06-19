#!/bin/bash
#SBATCH --job-name=brsim
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report V, Exp 3: the bridged-cliques probe on the SIMILARITY read-out model.
# EVAL-ONLY -- the similarity checkpoints already exist on HPC (Report IV families
# runs; NOT deleted -- see istruzioni error 34 correction). eval_bridged_cliques.py
# auto-detects the read-out from the state_dict, so we just point it at them.
#
# Question: Report IV showed the similarity read-out DOUBLES the distance reach
# (3^L -> 2*3^L). If the clique-size "node budget" (~6-7 in the linear base, sec 5.2)
# is the SAME capacity expressed on dense graphs, similarity should push the bridged
# cross-block knee from ~6-7 to ~13-14. If the knee does NOT move, budget and
# distance-reach are different mechanisms. Either way it is decisive and ~free.
#
# The MIXED model is the clean test (dense cliques in-distribution); ER is the
# confounded cross-check (degree-heuristic OOD, read only as a sanity check).
#
#   sbatch scripts/eval_bridged_similarity.sbatch
# Then locally:  git pull ; python plot_bridged_cliques.py   (extend it for this dir)

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/report5/bridged_similarity"
NG=2000

run_one () {   # $1 = checkpoint, $2 = tag
  if [ -f "$1" ]; then
    echo "== $2  ($1) =="
    python eval_bridged_cliques.py --checkpoint "$1" --output_dir "$OUT_ROOT/$2" --n_graphs "$NG"
  else
    echo "-- skip $2 (no checkpoint at $1)"
  fi
}

# n20 similarity: mixed (clean, 8 seeds) + er (cross-check, 8 seeds)
for S in 1000 2000 3000 4000 5000 6000 7000 8000; do
  run_one "runs/families_n20/n20_mixed_roberta_similarity_lam0_seed${S}/last.pt" "n20_mixed_sim_seed${S}"
  run_one "runs/families_n20/n20_er_roberta_similarity_lam0_seed${S}/last.pt"    "n20_er_sim_seed${S}"
done
# n40 similarity: mixed + er (4 seeds) -- the decisive range (clique size up to 20)
for S in 1000 2000 3000 4000; do
  run_one "runs/families_n40/n40_mixed_roberta_similarity_lam0_seed${S}/last.pt" "n40_mixed_sim_seed${S}"
  run_one "runs/families_n40/n40_er_roberta_similarity_lam0_seed${S}/last.pt"    "n40_er_sim_seed${S}"
done

echo "bridged-cliques (similarity read-out) eval done -> $OUT_ROOT"
