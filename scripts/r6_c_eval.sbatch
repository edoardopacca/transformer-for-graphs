#!/bin/bash
#SBATCH --job-name=r6ceval
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report VI, Thread C -- iterated bridged cliques (EVAL-ONLY) on the clean bridged+split
# base checkpoints trained by r6_c_train.sbatch. Two questions, both kept WITHIN the
# distance capacity 3^L=9 so any failure is "unseen composition", not the distance wall:
#
#   C.1 chained bridged cliques (eval_clique_chain.py): does the single-bridge "iff"
#       decision COMPOSE across a chain clique-bridge-clique-...? Per-link / per-gap
#       cross propagation + intact-vs-broken discrimination, swept over (n_cliques,
#       clique_size). Cells with diameter > 9 are auto-skipped. Also a chained-DENSE-ER-
#       block variant (the held-out block type, ties C.1 to C.2).
#   C.2 thick bridges + dense blocks (eval_bridged_cliques.py): the two-block sweep with
#       bridge_width in {1,2,3} (redundant hand-off, ties to Thread A parallelism) and the
#       held-out dense ER blocks (single bridge). Clique-size sweep vs the MP/BFS/DFS oracles.
#
# Eval-only -> no retrain. ~8 checkpoints x ~6 conditions; medium partition (the 1h10 short
# cap is tight here). Re-runnable and additive; eval_* SKIP missing checkpoints silently
# (err. 43) -> read the .out ("-- skip ...") and COUNT the json before trusting the result.
#
#   sbatch scripts/r6_c_eval.sbatch
# Optionally gate on the training (afterany, not afterok -- err. 44):
#   sbatch --dependency=afterany:<trainJobId> scripts/r6_c_eval.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

NG_CHAIN=400
NG_BRIDGE=2000

find_ckpt () { for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done; echo ""; }

for N in 20 40; do
  for S in 1000 2000 3000 4000; do
    CKPT="$(find_ckpt "runs/report6/c_train/n${N}_bridged+split_roberta_linear_lam0_seed${S}/last.pt")"
    if [ -z "$CKPT" ]; then echo "-- skip n${N} seed${S} (no checkpoint)"; continue; fi
    echo "=== n${N} seed${S}  <-  $CKPT ==="

    # C.1 chained complete cliques (single-edge bridges)
    python eval_clique_chain.py --checkpoint "$CKPT" \
        --output_dir "runs/report6/clique_chain/n${N}_bridgedsplit_seed${S}" \
        --n_graphs "$NG_CHAIN" --block clique --bridge_width 1
    # C.1/C.2 chained DENSE ER blocks (held-out block type; beyond-cap cells auto-skipped)
    python eval_clique_chain.py --checkpoint "$CKPT" \
        --output_dir "runs/report6/clique_chain_er/n${N}_bridgedsplit_seed${S}" \
        --n_graphs "$NG_CHAIN" --block er --bridge_width 1

    # C.2 thick bridges: two cliques, bridge_width 1/2/3
    for W in 1 2 3; do
      python eval_bridged_cliques.py --checkpoint "$CKPT" \
          --output_dir "runs/report6/thick_bridges/n${N}_cliques_bw${W}_seed${S}" \
          --n_graphs "$NG_BRIDGE" --family cliques --bridge_width "$W"
    done
    # C.2 held-out dense ER blocks, single bridge
    python eval_bridged_cliques.py --checkpoint "$CKPT" \
        --output_dir "runs/report6/thick_bridges/n${N}_blocks_bw1_seed${S}" \
        --n_graphs "$NG_BRIDGE" --family blocks --bridge_width 1
  done
done

echo "Thread C eval done -> runs/report6/{clique_chain,clique_chain_er,thick_bridges}/   (count the json!)"
