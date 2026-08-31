#!/bin/bash
#SBATCH --job-name=r10redmix
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-31, Edoardo: train directly on graphs that already contain a genuine
# long-range redundant-route pair, instead of testing an already-trained plain-disjoint-path
# model OOD on one (eval_n60_multipath_v2.py Constructions A-D). New family "redundant_mix"
# (experiments2/train_families_n20.py): a fixed 50/50 mix of exactly two shapes, both with
# the redundant pair's true distance safely past the 2*3^L=18 wall in EVERY sample:
#   (1) stitched theta graph (disjoint (20,20,20) split + 4 stitching edges -- the SAME
#       shape as Construction B), hub pair at true distance 19, 3 routes of 19/21/21 edges.
#   (2) two parallel 20-edge routes between a fresh s,t pair (true distance 20), remainder
#       auto-filled into one separate disjoint chain.
# Same architecture/read-out/step budget as every other n=60 training in this project:
# RoBERTa, similarity read-out, normalized-ReLU attention, 1e6 steps, batch 1000, seed 1000
# to start. Smoke-tested locally (n=60 -- this family needs the exact real canvas size,
# unlike most others -- 20 real steps, CPU): runs end-to-end, checkpoint saves correctly,
# new data.py generator generate_stitched_theta_graph verified to produce bit-identical
# adjacency to eval_n60_multipath_v2.py's build_construction_b (networkx-checked: hub
# distance 19, hub degree 3, 61 edges).
#
# --time/--partition: medium_gpuh200 (cap 6h10) instead of long_gpuh200 -- the K=3 n=60
# trainings of this exact size/step budget actually took ~4.5h (scripts/
# r10_n60_pathunion_k3_seed*.sbatch), so this should schedule faster on the shorter-cap,
# less congested partition (istruzioni.md partition guidance). Verify the partition still
# exists with `sinfo -o "%P %G %l %D"` before trusting this -- names can drift after cluster
# maintenance (istruzioni.md error #9).
#   sbatch scripts/r10_n60_redundant_mix_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report10/n60_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families redundant_mix --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_redundant_mix_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet"
echo "checkpoint at $CKPT"
