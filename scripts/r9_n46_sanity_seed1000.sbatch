#!/bin/bash
#SBATCH --job-name=r9n46s1
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- n=46 sanity check, seed 1000 (one of two INDEPENDENT sbatch files, one per
# seed, on purpose: this is a preliminary check of whether the canvas size n=46 reproduces
# the path-endpoint completion signal (Report VI/VII/VIII) before committing every other
# Report IX training to n=46, so it is launched and read as its own small experiment, not
# folded into a bigger array). Training is IDENTICAL to the Report VIII / Report VI Thread
# A1 disjoint-paths condition -- same family (path_union: online unions of 1..4 disjoint
# paths), same architecture (RoBERTa-faithful, L=2, single head, d_model=512), same
# similarity read-out, same step/sample budget (1e6 steps, batch 1000) -- the ONLY thing
# that changes is the canvas size, n=40 -> n=46.
#
# After training, auto-evaluates the split sweep (eval_asym_chains.py) on the resulting
# checkpoint: the direct behavioural check for the endpoint-completion signature (a small
# split should resolve the long component far past the distance wall; a balanced split
# should not, with the same weights) -- exactly the shape Report VI Thread B / Report VII
# established at n=40.
#
#   sbatch scripts/r9_n46_sanity_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report9/n46_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families path_union --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_path_union_roberta_similarity_lam0_seed${SEED}/last.pt"
echo "auto-eval asym-chains split sweep (endpoint-completion check): $CKPT"
python eval_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/asym_chains_n46/n46_pathunion_seed${SEED}" \
    --n_graphs 300
