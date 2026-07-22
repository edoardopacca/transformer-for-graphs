#!/bin/bash
#SBATCH --job-name=r8edgechain
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VIII -- the causal edge-ablation contribution (istruzioni.md sec 15
# update, report/8 sec:battery-contrib), on the SAME condition as the rest of
# report/8 sec:battery: n40, path_union (disjoint-paths) training, similarity
# read-out, the chain (two-chain) topology. This is the direct causal
# complement to the existing Jacobian-based node-to-node contribution
# (mechanistic_asym_chains.py::exact_contribution, kept as-is and reframed as
# an internal-sensitivity diagnostic, not replaced) -- see
# edge_ablation_contribution.py's docstring for the full design.
#
# Checkpoints (existing, no training):
#   runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed{S}/last.pt
#
# CPU-only, eval-only, NO backward pass at all (pure forward-pass ablation --
# cheaper than mechanistic_asym_chains.py's Jacobian probe, hence the shorter
# --time). Default splits (1,4,7,8,10,11,12,13,14,17,20) and n_graphs=64
# match the rest of this report for a direct comparison.
#
# Launch (single command, 4 seeds, medium_cpu has 8 nodes):
#   sbatch scripts/r8_edge_contrib_n40_pathunion.sbatch

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEEDS=(1000 2000 3000 4000)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed${S}/last.pt"
TAG="n40_pathunion_seed${S}"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

echo "== edge_ablation_contribution.py --topology chain (seed $S) =="
python edge_ablation_contribution.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report8/edge_contrib/${TAG}" \
  --topology chain \
  --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: edge_ablation_contribution.py"; exit 1; }
echo "done seed $S"
