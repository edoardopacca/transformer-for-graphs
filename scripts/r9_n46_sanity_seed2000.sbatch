#!/bin/bash
#SBATCH --job-name=r9n46s2
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- n=46 sanity check, seed 2000. Twin of scripts/r9_n46_sanity_seed1000.sbatch
# (see that file for the full rationale): identical training, only the seed differs. Kept
# as a SEPARATE sbatch file on purpose (two independent single-task jobs, not a 2-task
# array), so each can be inspected/resubmitted on its own.
#
#   sbatch scripts/r9_n46_sanity_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=2000
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
