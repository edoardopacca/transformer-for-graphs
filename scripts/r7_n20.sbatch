#!/bin/bash
#SBATCH --job-name=r7n20
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --array=0-7
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VII -- a THIRD canvas size for the mechanistic audit already done at
# n=40 (scripts/r7_exact_contribution.sbatch, scripts/r7_er_n40.sbatch) and at
# n=64 (sec:res-n64, Report VI Thread-A checkpoints): n=20, both training
# distributions (path_union and ER), as a correctness/robustness check that
# the exact-contribution approach and its findings are not an artefact of one
# particular canvas size. NB (istruzioni.md, Report VI Thread B note): at
# n=20 every within-component distance stays <= the capacity 3^L=9 even at
# the most balanced split (10,10) -- so unlike n=40/n=64 the behavioural wall
# may not bite the same way here; that is exactly the point of testing it,
# not a reason to skip it.
#
# Four pieces per (distribution, seed), all eval-only (no training, no GPU):
#   1. mechanistic_asym_chains.py -- split sweep (a=1..10) + relabel + read-out
#      geometry + read-out decomposition + attention probe (exact
#      node-to-node contribution, --attn_n_graphs/--contrib_n_graphs both 64
#      -- fixed from the start here, unlike the first n40 path_union run
#      where attn_n_graphs was left at its 40 default; see istruzioni sec 14.7).
#   2. mechanistic_heatmaps.py -- attention-score/weight heatmaps, exact
#      contribution, over ALL FIVE representative splits (1,4,7,8,10) rather
#      than a hardcoded pair: at n=20 it is not yet known from behaviour
#      alone which split is the cleanest "solved" vs "failed" contrast (the
#      plotting script's --pair can select the right two once the sweep
#      numbers are in; this eval step just needs to cover the candidates).
#   3. eval_three_way_split.py -- the falsification test (default small_lens
#      already clipped to n-2=18 by the script itself, no override needed).
#   4. ablation_asym_chains.py -- layer-by-layer (attention/MLP) ablations.
#
# CPU-only, BLAS multi-threaded across --cpus-per-task (this is dense
# matmuls + autograd, not model training). n=20 is roughly half the query
# rows of n=40 for the same d_model=512, so each piece should be noticeably
# cheaper than its n=40 counterpart; --time is left the same for margin.
#
# Launch (single command, 8 tasks = 2 distributions x 4 seeds; medium_cpu has
# 8 nodes, so this can run alongside the other r7_*.sbatch jobs if they are
# still going -- SLURM queues whatever does not fit):
#   sbatch scripts/r7_n20.sbatch

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

find_ckpt () {
  for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo ""
}

SEEDS=(1000 2000 3000 4000)
# index 0-3 -> path_union seeds 1000..4000 ; index 4-7 -> ER seeds 1000..4000
if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
  S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
  TAG="n20_pathunion_seed${S}"
  CKPT="$(find_ckpt "runs/report6/a1_train/n20_path_union_roberta_linear_lam0_seed${S}/last.pt")"
else
  IDX=$((SLURM_ARRAY_TASK_ID - 4))
  S=${SEEDS[$IDX]}
  TAG="n20_er_seed${S}"
  CKPT="$(find_ckpt \
    "runs/report3/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt" \
    "runs/repro_paper_n20_roberta/n20_p008_unrestricted_seed${S}/best.pt")"
fi

if [ -z "$CKPT" ]; then
  echo "no checkpoint found for $TAG"
  exit 1
fi
echo "tag=$TAG checkpoint=$CKPT"

echo "== mechanistic_asym_chains.py (Tier-1 + attention probe, exact contribution) =="
FAIL=0

python mechanistic_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/mechanistic/${TAG}" \
  --attn_n_graphs 64 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (heatmap probe, exact contribution, all 5 representative splits) =="
python mechanistic_heatmaps.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/heatmaps/${TAG}" \
  --splits 1 4 7 8 10 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== eval_three_way_split.py (falsification test) =="
python eval_three_way_split.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/three_way/${TAG}"
[ $? -ne 0 ] && { echo "FAILED: eval_three_way_split.py"; FAIL=1; }

echo "== ablation_asym_chains.py (layer ablations) =="
python ablation_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/ablation/${TAG}"
[ $? -ne 0 ] && { echo "FAILED: ablation_asym_chains.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED for $TAG -- see FAILED markers above"
  exit 1
fi
echo "done $TAG"
