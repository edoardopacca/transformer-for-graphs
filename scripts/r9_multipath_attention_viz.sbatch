#!/bin/bash
#SBATCH --job-name=r9mpattn
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- prof's request (2026-07-27 email): visualize the real attention scores
# for the multipath/parallel-paths experiments (Report VI Thread A), comparing k=1
# (single route, fails beyond capacity) against k>1 (parallel routes, rescues the
# connection). Eval-only, forward passes only -> CPU, existing checkpoints, NO
# training. Three conditions, so the comparison she asked for ("bisognerebbe
# confrontare con training sugli stessi grafi") is direct:
#   (a) ER-trained, n=64  -- never saw ANY path structure in training (Report VI
#       Thread A.1's cleanest "never saw multipath" condition).
#   (b) path_union-trained, n=40 -- saw single long paths but never >1 route between
#       two terminals (Report VI Thread A.1's other "never saw multipath" condition).
#   (c) multipath-trained, n=40, k=2/ell=13 -- trained DIRECTLY on the multipath
#       stream at this (k,ell) (Report VI Thread A.2), the in-distribution reading.
# ell=13 throughout (beyond the 3^L=9 capacity, matching Report VI's a1_profile_far
# condition, where the rescue effect is cleanest). Same seed (1000) for all three;
# --find_ckpt-style existence check (mirrors scripts/r6_a1_eval.sbatch) since these
# are pre-existing Report VI checkpoints, not produced by this job.
#   sbatch scripts/r9_multipath_attention_viz.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/report9/heatmaps_multipath"

find_ckpt () {
  for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo ""
}

FAIL=0

echo "== (a) ER-trained, n=64 (never saw any path structure) =="
CKPT="$(find_ckpt \
  runs/report6/a1_train/n64_er_roberta_linear_lam0_seed1000/last.pt)"
if [ -z "$CKPT" ]; then
  echo "-- skip (a): checkpoint not found"; FAIL=1
else
  python mechanistic_multipath_heatmaps.py --checkpoint "$CKPT" \
      --output_dir "$OUT_ROOT/n64_er_seed1000" --ks 1 2 3 4 --ell 13 --n_graphs 80
  [ $? -ne 0 ] && { echo "FAILED: (a) ER n64"; FAIL=1; }
fi

echo "== (b) path_union-trained, n=40 (single long paths, never >1 route) =="
CKPT="$(find_ckpt \
  runs/report6/a1_train/n40_path_union_roberta_linear_lam0_seed1000/last.pt)"
if [ -z "$CKPT" ]; then
  echo "-- skip (b): checkpoint not found"; FAIL=1
else
  python mechanistic_multipath_heatmaps.py --checkpoint "$CKPT" \
      --output_dir "$OUT_ROOT/n40_pathunion_seed1000" --ks 1 2 3 --ell 13 --n_graphs 80
  [ $? -ne 0 ] && { echo "FAILED: (b) path_union n40"; FAIL=1; }
fi

echo "== (c) multipath-trained, n=40, k=2/ell=13 (in-distribution reading) =="
CKPT="$(find_ckpt \
  runs/report6/multipath_train/n40_k2_ell13_clean_roberta_linear_seed1000/last.pt)"
if [ -z "$CKPT" ]; then
  echo "-- skip (c): checkpoint not found"; FAIL=1
else
  python mechanistic_multipath_heatmaps.py --checkpoint "$CKPT" \
      --output_dir "$OUT_ROOT/n40_multipath_k2ell13_seed1000" --ks 1 2 3 --ell 13 --n_graphs 80
  [ $? -ne 0 ] && { echo "FAILED: (c) multipath-trained n40"; FAIL=1; }
fi

if [ "$FAIL" -ne 0 ]; then
  echo "one or more conditions FAILED/missing -- see markers above (checkpoint paths "
  echo "may need updating if Report VI's runs/report6/{a1_train,multipath_train}/ layout changed)"
  exit 1
fi
echo "done: multipath attention visualisation (a)+(b)+(c)"
