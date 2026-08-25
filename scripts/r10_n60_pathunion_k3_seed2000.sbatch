#!/bin/bash
#SBATCH --job-name=r10pu3n60b
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- training (not read-out-only fine-tuning) on the exactly-3-components
# path_union distribution (every 3-way split-size combination), at a NEW canvas size
# n=60 (feeds a later multipath-generalisation question -- setup otherwise identical to
# every other Report VI-IX n46/n40 training: RoBERTa arch, similarity read-out,
# normalized-ReLU attention, 1e6 steps, batch 1000, single seed to start).
#
# New family "path_union_k3" (experiments2/train_families_n20.py), calls
# generate_path_union_graph(n, rng, max_paths=3, min_paths=3) as its own training
# distribution (this generator call already existed for finetune_readout_threeway_full.py's
# read-out-only fine-tuning stream; this is the first time it trains a whole trunk from
# scratch).
#
# --time/--partition: no historical n=60 timing exists yet. n=46 runs of this size/step
# budget take ~3.5-4h in practice despite a 16h SLURM time limit (see sacct history for
# r9grid3/4/5, r9splcyc5, r9gridc3/4/5, all ~3.5-4h). Attention cost scales ~(60/46)^2 ~= 1.7x
# n=46's, so even a pessimistic 3x-slower-shared-node scenario stays well inside 30h; using
# long_gpuh200 (3-day cap) rather than the 1-day nude partition purely for margin, since a
# timeout before completion loses the run (history.json is written only at the end -- errore
# 4 in istruzioni.md). Revisit this budget once one n=60 run's actual wall time is known.
#   sbatch scripts/r10_n60_pathunion_k3_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=2000
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
