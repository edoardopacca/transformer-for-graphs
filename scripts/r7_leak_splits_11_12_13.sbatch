#!/bin/bash
#SBATCH --job-name=r7leakgap
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VII -- prof feedback on the PPT built from this report (istruzioni.md errore 62):
# the attention-leak-fraction figure (fig:r7attn / r7_attention_leak.png, "Figure 3") found a
# spike at a=10 but the next probed split jumped straight to a=14, leaving the shape of the
# spike (does it come back down at 11-13, or stay elevated?) unresolved. This job adds ONLY
# the missing splits a=11,12,13 to the existing attention probe for the PRIMARY condition
# (n40, disjoint-paths-trained -- the checkpoint set behind Figure 3 itself).
#
# mechanistic_asym_chains.py now MERGES attn_cache.npz with whatever splits were already
# cached instead of overwriting the file, so this does NOT recompute the expensive
# exact-contribution quantity for the 8 splits already done (1,4,7,8,10,14,17,20) -- only
# the 3 new ones -- and it is safe to point back at the SAME output directories.
#
# The other "twin" leak-fraction figures in this report (ER-trained at n40/n64/n20,
# path_union at n64/n20, both similarity conditions) have the same gap by construction
# (same default attn_splits in mechanistic_asym_chains.py before this fix) and should get
# the same treatment; see istruzioni.md errore 62 for the checkpoint list and the pattern
# to extend this script to them (just change CKPT/TAG below per condition).
#
# Launch:
#   sbatch scripts/r7_leak_splits_11_12_13.sbatch

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEEDS=(1000 2000 3000 4000)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT="runs/report6/a1_train/n40_path_union_roberta_linear_lam0_seed${S}/last.pt"
TAG="n40_pathunion_seed${S}"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

echo "== mechanistic_asym_chains.py (attention probe, splits 11,12,13 only) -- seed $S =="
python mechanistic_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/mechanistic/${TAG}" \
  --attn_splits 11 12 13 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; exit 1; }
echo "done seed $S"
