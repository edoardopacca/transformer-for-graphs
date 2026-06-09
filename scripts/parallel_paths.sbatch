#!/bin/bash
#SBATCH --job-name=parpaths
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Controlled distance-vs-paths probe on the existing n=64 reach checkpoints
# (L=1,2,3). At n=64 there is room for long paths (ell up to 15) with several
# parallel paths, so we can test past the 3^L capacity. Light: minutes per model.
#
#   sbatch scripts/parallel_paths.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

for d in runs/reach_depth/reach_n64_L*; do
  [ -f "$d/last.pt" ] || continue
  echo "== $d =="
  python analyze_parallel_paths.py --checkpoint "$d/last.pt" \
      --output_dir "$d/parallel_paths" \
      --path_lens 5 7 9 11 13 15 --max_paths 4 --size 400
done
echo "parallel-paths probe done"
