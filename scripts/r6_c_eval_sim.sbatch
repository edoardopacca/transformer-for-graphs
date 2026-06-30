#!/bin/bash
#SBATCH --job-name=r6cevalS
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report VI, Thread C -- SIMILARITY read-out variant (EVAL-ONLY). Same as r6_c_eval.sbatch but on
# the SIMILARITY bridged+split base checkpoints (r6_c_train_sim.sbatch), writing all json to
# SEPARATE *_sim/ dirs so nothing collides with the linear Thread-C results (which another chat
# is analysing). eval_clique_chain.py / eval_bridged_cliques.py auto-detect the read-out.
# Same two questions, both kept WITHIN distance capacity 3^L=9:
#   C.1 chained bridged cliques (does the single-bridge "iff" compose across a chain?)
#   C.2 thick bridges (bridge_width 1/2/3) + held-out dense ER blocks.
#
# Eval-only -> no retrain. ~8 checkpoints x ~6 conditions; medium partition. Re-runnable and
# additive; eval_* SKIP missing checkpoints silently (err. 43) -> read the .out and COUNT the json.
#   sbatch scripts/r6_c_eval_sim.sbatch
# Gate on the similarity training (afterany, not afterok -- err. 44):
#   sbatch --dependency=afterany:<trainSimJobId> scripts/r6_c_eval_sim.sbatch

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
    CKPT="$(find_ckpt "runs/report6/c_train/n${N}_bridged+split_roberta_similarity_lam0_seed${S}/last.pt")"
    if [ -z "$CKPT" ]; then echo "-- skip n${N} seed${S} (no similarity checkpoint)"; continue; fi
    echo "=== n${N} seed${S}  <-  $CKPT ==="

    # C.1 chained complete cliques (single-edge bridges)
    python eval_clique_chain.py --checkpoint "$CKPT" \
        --output_dir "runs/report6/clique_chain_sim/n${N}_bridgedsplit_seed${S}" \
        --n_graphs "$NG_CHAIN" --block clique --bridge_width 1
    # C.1/C.2 chained DENSE ER blocks (held-out block type; beyond-cap cells auto-skipped)
    python eval_clique_chain.py --checkpoint "$CKPT" \
        --output_dir "runs/report6/clique_chain_er_sim/n${N}_bridgedsplit_seed${S}" \
        --n_graphs "$NG_CHAIN" --block er --bridge_width 1

    # C.2 thick bridges: two cliques, bridge_width 1/2/3
    for W in 1 2 3; do
      python eval_bridged_cliques.py --checkpoint "$CKPT" \
          --output_dir "runs/report6/thick_bridges_sim/n${N}_cliques_bw${W}_seed${S}" \
          --n_graphs "$NG_BRIDGE" --family cliques --bridge_width "$W"
    done
    # C.2 held-out dense ER blocks, single bridge
    python eval_bridged_cliques.py --checkpoint "$CKPT" \
        --output_dir "runs/report6/thick_bridges_sim/n${N}_blocks_bw1_seed${S}" \
        --n_graphs "$NG_BRIDGE" --family blocks --bridge_width 1
  done
done

echo "Thread C (similarity) eval done -> runs/report6/{clique_chain_sim,clique_chain_er_sim,thick_bridges_sim}/   (count the json!)"
