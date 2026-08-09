#!/bin/bash
#SBATCH --job-name=r10ft3wayfull
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report 10 (paper prep, idea_paper_by_prof.tex sec 5.3) -- the BROADER companion to
# r10_finetune_threeway_splitchains_seed1000.sbatch. Instead of fine-tuning the
# read-out on the two named hard cells (15,15,16)/(7,15,24), this trains it on the
# FULL distribution of exactly-three-component path splits (every (s1,s2,s3), via
# generate_path_union_graph(min_paths=3,max_paths=3)), for 200000 steps (4x the
# targeted run) -- does exposure to the whole K=3 space let the read-out alone learn
# a genuinely GENERAL three-component rule, not just two overfit cells? Still only
# sim_scale/sim_bias trainable (2 parameters), so this stays cheap despite the much
# longer budget.
#
# Same "before" baselines as the targeted run (Report IX's already-committed evals
# on the untouched checkpoint); only the AFTER-fine-tuning versions are computed
# here, into runs/report10/.
#
#   sbatch scripts/r10_finetune_full3way_splitchains_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt"
TAG="n46_splitchains_seed1000_full3way"
CELLS="1,22,23 4,15,27 7,15,24 2,10,34 15,15,16"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

FAIL=0

echo "== (1) fine-tune ONLY sim_scale/sim_bias on the FULL K=3 split distribution =="
python finetune_readout_threeway_full.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/finetune_readout_full3way/${TAG}" \
    --finetune_steps 200000 --batch_size 500 --lr 1e-3 \
    --eval_every 2000 --eval_graphs_aggregate 300 --eval_graphs_named 300 --seed 1000
[ $? -ne 0 ] && { echo "FAILED: finetune_readout_threeway_full.py"; FAIL=1; }

FT_CKPT="runs/report10/finetune_readout_full3way/${TAG}/finetuned.pt"
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
