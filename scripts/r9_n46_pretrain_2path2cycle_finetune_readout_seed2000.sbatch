#!/bin/bash
#SBATCH --job-name=r9ft2
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=17:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- second seed of the pretrain-{split_chains,split_cycles}-then-fine-tune-
# only-the-readout condition (scripts/r9_n46_pretrain_2path2cycle_finetune_readout_
# seed1000.sbatch was seed 1000, currently the only seed in scratch_prof_experiments.tex
# Sec. 5, where the read-out-only fine-tune left K>=3 exact match at 0.000 throughout).
# Identical three-stage setup (pretrain full transformer on {split_chains, split_cycles};
# BEFORE eval on K=5,6,7 + own-family sanity; freeze trunk and fine-tune only
# sim_scale/sim_bias on a K in {3..6} stream; AFTER eval on the same K=5,6,7 + own-family
# sanity), seed 2000 only.
#   sbatch scripts/r9_n46_pretrain_2path2cycle_finetune_readout_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=2000
OUT_ROOT="runs/report9/n46_train"
TAG="n46_2path2cycle_seed2000"

echo "== (1) pretrain: full transformer on split_chains + split_cycles =="
python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_chains,split_cycles --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_chains+split_cycles_roberta_similarity_lam0_seed2000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "pretraining did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0

echo "== (own-family sanity, BEFORE fine-tuning) eval_asym_chains.py [chain] =="
python eval_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/asym_chains_n46/${TAG}_pre" --n_graphs 300
[ $? -ne 0 ] && { echo "FAILED: eval_asym_chains.py (pre)"; FAIL=1; }

echo "== (own-family sanity, BEFORE fine-tuning) mechanistic_asym_chains.py [cycle] =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}_pre_cycle" --topology cycle --attn_splits 4 23
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (pre, cycle)"; FAIL=1; }

echo "== (2) BEFORE fine-tuning: K=5,6,7 OOD eval (eval_multiway_split.py) =="
python eval_multiway_split.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/multiway_split/${TAG}_pre" --n_graphs 300
[ $? -ne 0 ] && { echo "FAILED: eval_multiway_split.py (pre)"; FAIL=1; }

echo "== (3) fine-tune ONLY the read-out on a K>=3 disjoint-path stream =="
python finetune_readout_only.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/finetune_readout/${TAG}" \
    --min_paths 3 --max_paths 6 --finetune_steps 3000 --batch_size 500 \
    --lr 1e-3 --eval_every 200 --eval_graphs 300 --seed "$SEED"
[ $? -ne 0 ] && { echo "FAILED: finetune_readout_only.py"; FAIL=1; }

FT_CKPT="runs/report9/finetune_readout/${TAG}/finetuned.pt"
if [ -f "$FT_CKPT" ]; then
  echo "== (own-family sanity, AFTER fine-tuning) eval_asym_chains.py [chain] =="
  python eval_asym_chains.py --checkpoint "$FT_CKPT" \
      --output_dir "runs/report9/asym_chains_n46/${TAG}_post" --n_graphs 300
  [ $? -ne 0 ] && { echo "FAILED: eval_asym_chains.py (post)"; FAIL=1; }

  echo "== (own-family sanity, AFTER fine-tuning) mechanistic_asym_chains.py [cycle] =="
  python mechanistic_asym_chains.py --checkpoint "$FT_CKPT" \
      --output_dir "runs/report9/mechanistic/${TAG}_post_cycle" --topology cycle --attn_splits 4 23
  [ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (post, cycle)"; FAIL=1; }

  echo "== AFTER fine-tuning: K=5,6,7 OOD eval (eval_multiway_split.py) =="
  python eval_multiway_split.py --checkpoint "$FT_CKPT" \
      --output_dir "runs/report9/multiway_split/${TAG}_post" --n_graphs 300
  [ $? -ne 0 ] && { echo "FAILED: eval_multiway_split.py (post)"; FAIL=1; }
else
  echo "FAILED: no fine-tuned checkpoint at $FT_CKPT, skipping post-finetune evals"
  FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG (pre + post fine-tune battery)"
