#!/bin/bash
#SBATCH --job-name=r10dirch
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-28/29, Edoardo: does the multipath-redundancy connectivity finding
# (paper_draft.tex Sec. 4.3 -- a distance>2*3^L pair gets rescued when several routes exist
# instead of one) generalise from undirected graph connectivity to a genuine DIRECTED
# reasoning-chain task, in the spirit of Abbe et al.'s "How Far Can Transformers Reason? The
# Globality Barrier and Inductive Scratchpad" (syllogism composition A=>B, B=>C |- A=>C)?
#
# n=40 (Edoardo, 2026-08-29: reduced from the original n=60 plan -- everything else
# unchanged, same n both for THIS training run and for the later redundant-chain eval).
#
# New family "directed_chain" (experiments2/train_families_n20.py): a single directed
# Hamiltonian path 0->1->...->39 (generate_directed_chain_graph, data.py), target is directed
# reachability (compute_directed_reachability_matrix, data.py -- reach[i,j]=1 iff j reachable
# from i forward-only; NOT symmetric, unlike every other family in this project).
#
# Read-out: --readout linear (NOT similarity). The similarity read-out's Z_ij=cos(h_i,h_j) is
# symmetric by construction (cos(h_i,h_j)=cos(h_j,h_i)), so it CANNOT represent a genuinely
# directed target (A=>B without B=>A) -- confirmed by inspecting model.py before starting this.
# The existing "linear" read-out (logits[i,j] = W_j . h_i + b_j, already the CLI default) is
# already asymmetric and needed no changes.
#
# CAPACITY: Edoardo is confident the wall is the plain 3^L=9 for this read-out (unlike the
# similarity read-out's doubled 2*3^L=18 -- Theorem 1 in the paper is specific to the cosine/
# ball-intersection argument and does not transfer here). The redundant-chain rescue test
# planned once this checkpoint is ready: accuracy on a target pair s,t joined by 1/2/3/4
# independent directed chains, each individually longer than 9 hops (route length 11 edges
# for 1-3 routes; 10 edges for 4 routes, since 4x11 does not fit in n=40 -- see
# eval_directed_chain_redundancy.py once written).
#
# Same setup as every other training in this project otherwise: RoBERTa arch, normalized-ReLU
# attention, 1e6 steps, batch 1000, single seed to start. Smoke-tested locally (n=12, 20 real
# steps, CPU): runs end-to-end, checkpoint saves correctly, labels verified non-symmetric.
#
# --time/--partition: mirrors scripts/r10_n60_pathunion_k3_seed1000.sbatch's n=60 budget
# (~4.5h actual wall time there); n=40 should be faster (smaller attention matrices), keeping
# a generous 20h cap purely for margin (history.json is only written at the end -- a timeout
# loses the whole run, istruzioni.md error #4).
#   sbatch scripts/r10_n40_directed_chain_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report10/n40_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families directed_chain --arch roberta --readout linear --lambda_lap 0 \
    --n_nodes 40 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n40_directed_chain_roberta_linear_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet (redundant-chain rescue eval TBD)"
echo "checkpoint at $CKPT"
