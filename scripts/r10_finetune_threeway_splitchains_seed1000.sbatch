#!/bin/bash
#SBATCH --job-name=r10ft3way
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report 10 (paper prep, idea_paper_by_prof.tex sec 5.3 "Generalization to three paths,
# finetuning??") -- targeted fine-tuning of ONLY the similarity read-out's scale/bias on
# the two three-way split-chain cells the prof singled out from Report IX Thread A.3:
#   (15,15,16) -- fully balanced, all three cuts fail, but every stagewise cosine block
#                 already LOOKS learned (within-component blocks much darker than cross).
#   (7,15,24)  -- cut(2,3) fails; also has pairs inside the 24-node component beyond the
#                 doubled wall 2*3^L=18.
# Unlike Report IX experiment 6 (finetune_readout_only.py, generic K in {3..6} path_union
# stream, did NOT recover K>=3), this fine-tuning stream is NAMED and NARROW: only these
# two cells. Much longer than the earlier 3000-step probe (no 30k-step run exists in this
# repo -- verified by grep before writing this script) since only 2 scalars are trainable,
# so it stays cheap.
#
# Reuses, as the "before" state, Report IX's ALREADY-COMMITTED evals on the untouched
# checkpoint (same weights, nothing to rerun): runs/report9/asym_chains_n46/
# n46_splitchains_seed1000 (own-family K=2 sweep), runs/report9/stagewise_threeway/
# n46_splitchains_seed1000 and runs/report9/heatmaps_threeway/n46_splitchains_seed1000
# (5-cell mechanistic battery, includes both target cells), runs/report9/
# threeway_splitchains/n46_splitchains_seed1000 (full 176-cell A.3 behavioural sweep).
# Only the AFTER-fine-tuning versions of these are computed here, into runs/report10/.
#
#   sbatch scripts/r10_finetune_threeway_splitchains_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt"
TAG="n46_splitchains_seed1000_threeway"
CELLS="1,22,23 4,15,27 7,15,24 2,10,34 15,15,16"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

FAIL=0

echo "== (1) fine-tune ONLY sim_scale/sim_bias on {(15,15,16),(7,15,24)} =="
python finetune_readout_threeway.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/finetune_readout_threeway/${TAG}" \
    --target_cells 15,15,16 7,15,24 \
    --finetune_steps 50000 --batch_size 500 --lr 1e-3 \
    --eval_every 1000 --eval_graphs 300 --seed 1000
[ $? -ne 0 ] && { echo "FAILED: finetune_readout_threeway.py"; FAIL=1; }

FT_CKPT="runs/report10/finetune_readout_threeway/${TAG}/finetuned.pt"
if [ -f "$FT_CKPT" ]; then

  echo "== (2) AFTER fine-tuning: own-family K=2 split-chains sweep (forgetting check) =="
  python eval_asym_chains.py --checkpoint "$FT_CKPT" \
      --output_dir "runs/report10/asym_chains_n46/${TAG}_post" --n_graphs 300
  [ $? -ne 0 ] && { echo "FAILED: eval_asym_chains.py (post)"; FAIL=1; }

  echo "== (3) AFTER fine-tuning: full 176-cell A.3 behavioural sweep =="
  python eval_threeway_splitchains.py --checkpoint "$FT_CKPT" \
      --output_dir "runs/report10/threeway_splitchains/${TAG}_post" --n_graphs 200
  [ $? -ne 0 ] && { echo "FAILED: eval_threeway_splitchains.py (post)"; FAIL=1; }

  echo "== (4) AFTER fine-tuning: attention heatmaps, 5 cells (same as Report IX A.3) =="
  python mechanistic_threeway_heatmaps.py --checkpoint "$FT_CKPT" \
      --output_dir "runs/report10/heatmaps_threeway/${TAG}_post" --cells $CELLS
  [ $? -ne 0 ] && { echo "FAILED: mechanistic_threeway_heatmaps.py (post)"; FAIL=1; }

  echo "== (5) AFTER fine-tuning: stagewise cosine geometry, 5 cells =="
  python stagewise_threeway.py --checkpoint "$FT_CKPT" \
      --output_dir "runs/report10/stagewise_threeway/${TAG}_post" --cells $CELLS --n_graphs 64
  [ $? -ne 0 ] && { echo "FAILED: stagewise_threeway.py (post)"; FAIL=1; }

else
  echo "FAILED: no fine-tuned checkpoint at $FT_CKPT, skipping post-finetune evals"
  FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG (fine-tune + full post-finetune battery; pre-finetune baselines already in runs/report9/)"
