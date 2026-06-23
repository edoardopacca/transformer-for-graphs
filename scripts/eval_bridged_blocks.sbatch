#!/bin/bash
#SBATCH --job-name=brblocks
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report V, OOD generalisation check for the train-on-bridged result (5.6):
# does the model that learned to propagate a single bridge across a dense CLIQUE
# also do it on a HELD-OUT bridge family it never trained on -- two dense ER
# BLOCKS (not complete cliques) joined by one edge? Same "carry one bridge across
# a dense region of c nodes" challenge, different internal structure.
#
#   - if the trained-on-bridged model stays flat ~1.0 across block size c
#       -> it learned a GENERAL propagation skill -> 5.6 was a data gap (soft).
#   - if it collapses with c (like the held-out base)
#       -> it only memorised the clique family -> the difficulty is real for new
#          dense-region shapes (weakens the data-prior reading).
#
# Eval-only on EXISTING checkpoints (no retrain). The matrix-power oracle is flat
# at 1.0 here too (cross distance <= ~5 << capacity 9), so this isolates
# propagation from distance, exactly like the bridged-clique probe.
# ~1-3 min per checkpoint.
#
#   sbatch scripts/eval_bridged_blocks.sbatch
# Then locally:  git pull   (and a plot/table once the json are in)
#
# error 43: eval_bridged_cliques.py SKIPS missing checkpoints silently -- always
# read the .out ("-- skip ...") and count the json before trusting the result.

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

NG=2000

run_one () {   # $1 = checkpoint, $2 = output_dir
  if [ -f "$1" ]; then
    echo "== $2  ($1) =="
    python eval_bridged_cliques.py --checkpoint "$1" --output_dir "$2" \
        --n_graphs "$NG" --family blocks
  else
    echo "-- skip $2 (no checkpoint at $1)"
  fi
}

for N in 20 40; do
  for S in 1000 2000 3000 4000; do
    # trained WITH bridged cliques in the stream (the 5.6 model)
    run_one "runs/report5/train_bridged_in_stream/n${N}_mixedbr_roberta_linear_lam0_seed${S}/last.pt" \
            "runs/report5/bridged_blocks_trained/n${N}_seed${S}"
    # held-out base (the 5.2 model, bridged NOT in training) -- the control
    run_one "runs/report5/base_n${N}/n${N}_mixed_roberta_linear_lam0_seed${S}/last.pt" \
            "runs/report5/bridged_blocks_heldout/n${N}_seed${S}"
  done
done

echo "bridged-BLOCKS eval done -> runs/report5/bridged_blocks_{trained,heldout}"
