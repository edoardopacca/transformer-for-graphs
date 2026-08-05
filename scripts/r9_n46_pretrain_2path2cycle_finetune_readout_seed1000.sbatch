#!/bin/bash
#SBATCH --job-name=r9ft1
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=17:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- prof's fine-tuning question (2026-07-27 email, verbatim): "per i
# cycles/paths, si potrebbe provare a trainare tutto il transformer su 2 paths e 2
# cycles, e poi fare qualche step di fine-tuning sulla distribuzione test (ad esempio,
# 3 o piu paths) solo del readout." Single job, three stages, one seed (exploratory):
#   (1) pretrain the FULL transformer on {split_chains, split_cycles} (a named mixture
#       of exactly the two 2-component families, principle sec 13.2 -- never K>=3);
#   (2) evaluate that pretrained checkpoint OOD on K=5,6,7 disjoint-path unions
#       (eval_multiway_split.py, the same construction as Thread A.4/sec:planA) --
#       the "before" baseline;
#   (3) freeze the WHOLE trunk and fine-tune ONLY the read-out (for similarity: the
#       two scalars sim_scale/sim_bias) for a few thousand steps on a K>=3 disjoint-
#       path stream (finetune_readout_only.py), then re-run the SAME K=5,6,7 eval on
#       the fine-tuned checkpoint -- the "after" comparison. Also re-runs the own-
#       family (2-chain, 2-cycle) sweeps on both checkpoints as a sanity check that
#       touching 2 scalars did not wreck in-distribution behaviour.
# n=46, similarity read-out, same architecture/step budget as every other Report IX
# training for the pretrain stage; the fine-tune stage is deliberately CHEAP (frozen
# trunk, 2 trainable scalars, 3000 steps) -- a probe, not a retrain.
#   sbatch scripts/r9_n46_pretrain_2path2cycle_finetune_readout_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report9/n46_train"
TAG="n46_2path2cycle_seed1000"

echo "== (1) pretrain: full transformer on split_chains + split_cycles =="
python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_chains,split_cycles --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_chains+split_cycles_roberta_similarity_lam0_seed1000/last.pt"
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
