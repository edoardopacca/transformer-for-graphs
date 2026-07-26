#!/bin/bash
#SBATCH --job-name=r9a3three2
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX, Thread A.3 -- SAME test as scripts/r9_a3_splitchains_threeway_test.sbatch, run on
# the OTHER n=46 seed-1000 checkpoint trained without ever seeing 3+ components: the
# "1chain+split_chains" mixture (this report's condition (4), 50/50 a single 46-node chain
# (k=1) or a random two-way split (k=2) -- never k=3 or more). Same eval_threeway_splitchains.py
# script, same 176 three-way size combinations (every partition of 46 into 3 positive parts),
# unchanged -- only the checkpoint differs, to see whether the split_chains-only checkpoint's
# Thread A.3 pattern (§16.13/§16.14 of istruzioni.md: whole-graph exact match always 0, failure
# concentrated in exactly one of the three pairwise cuts, a trivial size-1 component acting as
# the only reliable anchor) is a property of the split_chains-only training specifically, or
# also holds for a checkpoint that saw a richer (still <=2-component) mixture.
#
# Eval-only, CPU, same cost as the sibling script.
#
#   sbatch scripts/r9_a3_onetwopath_threeway_test.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_1chain+split_chains_roberta_similarity_lam0_seed1000/last.pt"
OUTDIR="runs/report9/threeway_splitchains/n46_onetwopath_seed1000"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

python eval_threeway_splitchains.py \
    --checkpoint "$CKPT" \
    --output_dir "$OUTDIR" \
    --n_graphs 200 \
    --min_size 1

echo "done: $OUTDIR/threeway_split.csv"
