#!/bin/bash
#SBATCH --job-name=r10mpv28000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-31, Edoardo: extend the n=60 multipath-redundancy eval
# (Constructions A/B/C/D, eval_n60_multipath_v2.py) to seed 8000, one of the 8 new K=3
# path-union training seeds (3000-10000) added to reach 10 seeds total, alongside the
# original seeds 1000/2000 (scripts/r10_n60_multipath_v2_seed{1000,2000}.sbatch). Identical
# eval, only the checkpoint/seed changes.
#   sbatch scripts/r10_n60_multipath_v2_seed8000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed8000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_n60_multipath_v2.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/multipath_v2/n60_pathunion_k3_seed8000" \
    --route_lens_a 20 21 20 --term_deg_a 3 --disjoint_sizes_b 20 20 20 \
    --n_graphs_behaviour 50 --n_repeats 5 --n_graphs_heatmap 64 \
    --seed 8000 --tag "seed 8000"
[ $? -ne 0 ] && { echo "FAILED: eval_n60_multipath_v2.py"; exit 1; }
echo "done: n60_pathunion_k3_seed8000 multipath v2"
