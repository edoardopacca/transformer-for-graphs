#!/bin/bash
#SBATCH --job-name=r8stage
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VIII -- Priority 1 of the "where is connectivity information
# combined?" thread (istruzioni.md sec 15, report/8 sec:question):
# stagewise_diagnostics.py, on the SAME condition as report/8 sec:battery
# (n40, disjoint-paths/path_union training, similarity read-out).
#
# Forward-pass only (no backward pass at all, unlike mechanistic_asym_chains.py's
# exact-Jacobian contribution) -- captures every intermediate state of the
# 2-layer trunk (H^(0), AttnOut/Hattn/FFNOut/H per layer), the layerwise cosine
# geometry the similarity read-out actually uses, an "intermediate read-out"
# accuracy probe at each stage, the far-margin, and the per-sub-block logit
# deltas (attn1/mlp1/attn2/mlp2) with their category means. See
# stagewise_diagnostics.py's docstring for the full design (it implements
# sections 1-5 of the 12-section brief the user's spec laid out; sections 6-10
# -- Q/K/V/attention-output norms, causal short-to-long masking, stagewise edge
# ablation, chord control -- are priorities 2/3, not run by this job).
#
# Checkpoints (existing, similarity read-out, no training):
#   runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed{S}/last.pt
#
# CPU-only, eval-only, no backward pass -- cheap (comparable to
# edge_ablation_contribution.py, which completed in under a minute per seed
# on this partition), hence the short --time.
#
# Launch (single command, 4 seeds, medium_cpu has 8 nodes):
#   sbatch scripts/r8_stagewise_n40_pathunion.sbatch

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

echo "== stagewise_diagnostics.py (seed $S) =="
python stagewise_diagnostics.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report8/stagewise/${TAG}" \
  --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py"; exit 1; }
echo "done seed $S"
