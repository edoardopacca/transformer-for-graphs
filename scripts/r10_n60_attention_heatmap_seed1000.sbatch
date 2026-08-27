#!/bin/bash
#SBATCH --job-name=r10attn1000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- real attention-weight heatmaps (layer 1 and layer 2) for the disjoint (20,20,20)
# split (report/10 Figure 16) and the two leaf-free multipath constructions A/B (Figure 17),
# n=60 K=3 (path_union_k3) checkpoint, seed 1000. New script eval_n60_attention_heatmap.py
# (smoke-tested locally on a random-weight n=60 checkpoint).
#   sbatch scripts/r10_n60_attention_heatmap_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed1000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_n60_attention_heatmap.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/attention_heatmap/n60_pathunion_k3_seed1000" \
    --n_graphs 64 --route_lens_a 20 21 20 --term_deg_a 3 --disjoint_sizes 20 20 20 \
    --seed 1000 --tag "seed 1000"
[ $? -ne 0 ] && { echo "FAILED: eval_n60_attention_heatmap.py"; exit 1; }
echo "done: n60_pathunion_k3_seed1000 attention heatmaps"
