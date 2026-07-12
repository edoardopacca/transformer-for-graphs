#!/bin/bash
#SBATCH --job-name=r7twabfix
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VII -- FIX-UP job. scripts/r7_er_n40.sbatch and scripts/r7_n20.sbatch both
# COMPLETED on SLURM, but their eval_three_way_split.py / ablation_asym_chains.py
# steps silently failed: those two scripts (and the data.py::generate_three_way_split_graph
# function eval_three_way_split.py needs) had never been pushed to git before this
# fix, so `python eval_three_way_split.py ...` errored with "no such file" on HPC --
# and because neither sbatch used `set -e`, the job kept going and reported
# COMPLETED anyway. mechanistic_asym_chains.py / mechanistic_heatmaps.py (the
# expensive exact-contribution steps) DID work correctly and do not need
# re-running -- this job re-runs ONLY the two fast, purely-behavioural steps that
# were missing, for the 12 checkpoints affected (n40 ER x4, n20 path_union x4,
# n20 ER x4). Cheap and CPU-only -- short_cpu is enough (no exact_contribution
# here, just ordinary forward passes).
#
# Launch (single command, 12 tasks; short_cpu has 8 nodes so most run at once):
#   sbatch scripts/r7_threeway_ablation_fix.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

find_ckpt () {
  for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo ""
}

SEEDS=(1000 2000 3000 4000)
IDX=$((SLURM_ARRAY_TASK_ID % 4))
GROUP=$((SLURM_ARRAY_TASK_ID / 4))
S=${SEEDS[$IDX]}

case "$GROUP" in
  0)
    TAG="n40_er_seed${S}"
    CKPT="$(find_ckpt \
      "runs/report4/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt" \
      "runs/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt")"
    ;;
  1)
    TAG="n20_pathunion_seed${S}"
    CKPT="$(find_ckpt "runs/report6/a1_train/n20_path_union_roberta_linear_lam0_seed${S}/last.pt")"
    ;;
  2)
    TAG="n20_er_seed${S}"
    CKPT="$(find_ckpt \
      "runs/report3/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt" \
      "runs/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt")"
    ;;
esac

if [ -z "$CKPT" ]; then
  echo "no checkpoint found for $TAG"
  exit 1
fi
echo "tag=$TAG checkpoint=$CKPT"

echo "== eval_three_way_split.py (falsification test) =="
python eval_three_way_split.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/three_way/${TAG}"
TW_STATUS=$?

echo "== ablation_asym_chains.py (layer ablations) =="
python ablation_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/ablation/${TAG}"
AB_STATUS=$?

if [ "$TW_STATUS" -ne 0 ] || [ "$AB_STATUS" -ne 0 ]; then
  echo "FAILED: three_way_status=$TW_STATUS ablation_status=$AB_STATUS"
  exit 1
fi
echo "done $TAG"
