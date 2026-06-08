#!/bin/bash
#SBATCH --job-name=fam20
#SBATCH --account=3352759
#SBATCH --partition=long_gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Families experiment (Report IV), n=20, RoBERTa, 1M steps online.
# Each array task = ONE condition; after training it auto-runs eval_families.py
# on its own checkpoint, so results appear run-by-run.
#
# The RUNS table is ORDERED BY SEED: launch one seed-block at a time, e.g.
#   sbatch --array=0-9%4   scripts/train_families.sbatch   # seed 1000 (+ lambda sweep)
#   sbatch --array=10-15%4 scripts/train_families.sbatch   # seed 2000
#   sbatch --array=16-21%4 scripts/train_families.sbatch   # seed 3000   ... etc
# %4 respects the 4-concurrent-GPU cap. Indices per seed are printed by:
#   sbatch --array=0 scripts/train_families.sbatch         # (then read the .out)
#
# Arms per seed: {er,mixed} x {linear lam0, similarity lam0, similarity lam*}.
# Seed 1000 additionally sweeps similarity lambda in {1e-4,1e-3,1e-2} to fix lam*.
# 'er linear 0' for seeds 1000-8000 REUSES the existing Report-III roberta
# checkpoints (eval only); seeds 9000-13000 have none, so they train.

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf

cd $SLURM_SUBMIT_DIR
OUT_ROOT="runs/families_n20"
STEPS=1000000
LAMSTAR=1e-3                 # chosen from analysis + n=64 evidence; verified by the seed-1000 sweep
EXISTING_ROOT="runs/repro_paper_n20_roberta"

# ---- build the ordered RUNS table: "SET READOUT LAMBDA SEED ACTION" ----------
RUNS=()
# seed 1000: full lambda sweep
RUNS+=("er linear 0 1000 eval")
for RL in "similarity 0" "similarity 1e-4" "similarity 1e-3" "similarity 1e-2"; do RUNS+=("er $RL 1000 train"); done
RUNS+=("mixed linear 0 1000 train")
for RL in "similarity 0" "similarity 1e-4" "similarity 1e-3" "similarity 1e-2"; do RUNS+=("mixed $RL 1000 train"); done
# production seeds with existing er-linear checkpoints (eval reuse)
for S in 2000 3000 4000 5000 6000 7000 8000; do
  RUNS+=("er linear 0 $S eval")
  RUNS+=("er similarity 0 $S train")
  RUNS+=("er similarity $LAMSTAR $S train")
  RUNS+=("mixed linear 0 $S train")
  RUNS+=("mixed similarity 0 $S train")
  RUNS+=("mixed similarity $LAMSTAR $S train")
done
# new seeds (no existing checkpoint -> train er-linear too)
for S in 9000 10000 11000 12000 13000; do
  RUNS+=("er linear 0 $S train")
  RUNS+=("er similarity 0 $S train")
  RUNS+=("er similarity $LAMSTAR $S train")
  RUNS+=("mixed linear 0 $S train")
  RUNS+=("mixed similarity 0 $S train")
  RUNS+=("mixed similarity $LAMSTAR $S train")
done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS table (${#RUNS[@]} tasks):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read SET RO LAM SEED ACTION <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> set=$SET readout=$RO lambda=$LAM seed=$SEED action=$ACTION"

if [ "$ACTION" = "eval" ]; then
  CKPT="$EXISTING_ROOT/n20_p008_unrestricted_seed${SEED}/best.pt"
  OUTD="$EXISTING_ROOT/n20_p008_unrestricted_seed${SEED}/families"
  echo "EVAL-ONLY (reuse existing ckpt): $CKPT"
  python eval_families.py --checkpoint "$CKPT" --output_dir "$OUTD" --size 4000
  exit $?
fi

# training run + automatic eval on its own checkpoint
python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families "$SET" --arch roberta --readout "$RO" --lambda_lap "$LAM" \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

# match train_families_n20.py's run-name tag exactly: f"lam{lambda:g}" (1e-3 -> lam0.001)
LAMTAG="lam$(python -c "print(format(float('$LAM'),'g'))")"
RUN_NAME="n20_${SET}_roberta_${RO}_${LAMTAG}_seed${SEED}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
echo "auto-eval: $RUN_DIR/last.pt"
python eval_families.py --checkpoint "$RUN_DIR/last.pt" --output_dir "$RUN_DIR/families" --size 4000
