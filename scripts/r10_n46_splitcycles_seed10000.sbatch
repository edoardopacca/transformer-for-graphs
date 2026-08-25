#!/bin/bash
#SBATCH --job-name=r10splcyc10
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX/X -- split_cycles-only n=46, seed 10000 -- extending the continuous two-cycle
# condition from 5 to 10 seeds (istruzioni.md seed policy: conditions with substantial
# seed-to-seed variability -- the split_cycles seed lottery, Report IX Thread B -- go to 10
# seeds so the solving fraction can be quantified rather than described anecdotally). Same
# family/architecture/read-out/step budget/battery as seed1000-5000
# (scripts/r9_n46_splitcycles_seed5000.sbatch). eval_asym_chains.py is CHAIN-only so it is
# skipped, as in every other split_cycles seed script.
#   sbatch scripts/r10_n46_splitcycles_seed10000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=10000
OUT_ROOT="runs/report9/n46_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_cycles --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_cycles_roberta_similarity_lam0_seed10000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0
TAG="n46_splitcycles_seed10000"

echo "== mechanistic_asym_chains.py (sweep+logit, topology=cycle) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" --attn_splits 4 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (sweep+logit, topology=cycle)"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores, topology=cycle) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py (attention scores, topology=cycle)"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry, topology=cycle) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 23 --n_graphs 64 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py (layerwise cosine geometry, topology=cycle)"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
