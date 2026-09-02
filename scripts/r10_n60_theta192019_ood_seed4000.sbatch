#!/bin/bash
#SBATCH --job-name=r10t1920ood4000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-01, Edoardo: OOD test of the path_union_k3 seed 4000 checkpoint
# (trained on random 3-way splits, never on this exact shape) against theta_19_20_19
# (routes 19/20/20 + 2 isolated nodes) -- new script eval_theta192019_ood.py, smoke-tested
# locally on a random-weight n=60 checkpoint (mask sizes verified: main_other=1652,
# cross=116, isolated_pair=1, all matching expected C(n,2) counts).
#   sbatch scripts/r10_n60_theta192019_ood_seed4000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed4000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_theta192019_ood.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/theta192019_ood/n60_pathunion_k3_seed4000" \
    --n_graphs 300 --seed 4000
[ $? -ne 0 ] && { echo "FAILED: eval_theta192019_ood.py"; exit 1; }
echo "done: seed 4000"
