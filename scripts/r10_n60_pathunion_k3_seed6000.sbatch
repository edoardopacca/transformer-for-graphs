#!/bin/bash
#SBATCH --job-name=r10pu3n60
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-29, Edoardo: 8 more seeds (3000-10000) of the same K=3 path-union
# training at n=60 already done for seeds 1000/2000
# (scripts/r10_n60_pathunion_k3_seed{1000,2000}.sbatch), to reach 10 seeds total for the
# multipath-redundancy results (eval_n60_multipath_v2.py, Constructions A/B/C/D). Identical
# setup, only the seed changes: RoBERTa arch, similarity read-out, normalized-ReLU
# attention, family "path_union_k3" (generate_path_union_graph(n, rng, max_paths=3,
# min_paths=3)), 1e6 steps, batch 1000.
#
# --time: seeds 1000/2000 actually took ~4.5h each (confirmed, not estimated like the
# original run's 30h placeholder) -- keeping a generous 20h cap purely for margin on a
# shared cluster (history.json is only written at the end -- a timeout loses the whole run,
# istruzioni.md error #4).
#   sbatch scripts/r10_n60_pathunion_k3_seed6000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=6000
OUT_ROOT="runs/report10/n60_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families path_union_k3 --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_path_union_k3_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet (multipath eval TBD)"
echo "checkpoint at $CKPT"
