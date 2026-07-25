#!/bin/bash
#SBATCH --job-name=r9stgrfr
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report IX -- refresh EXISTING stagewise_geometry.npz files so they carry the checkpoint's
# scale/bias (stagewise_diagnostics.py now saves these -- see the code change in this
# session). Purely a re-run of an already-cheap, backward-pass-free script (~6-7s/seed,
# confirmed in Report VIII): no new experiment, just regenerating data the plotting code now
# needs. Overwrites the SAME output directories (identical seed/splits/n_graphs as the
# original runs, so metrics/margins/deltaz CSVs come out unchanged -- only the npz gains
# scale/bias). CPU-only, no GPU needed (same as every other stagewise/mechanistic run).
#
#   sbatch scripts/r9_stagewise_rescale_refresh.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

# task -> (checkpoint, output_dir, splits, topology)
declare -a CKPTS OUTDIRS SPLITS TOPOS
CKPTS[0]="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed1000/last.pt"
OUTDIRS[0]="runs/report8/stagewise/n40_pathunion_seed1000"
SPLITS[0]="4 7 8 10 14 20"; TOPOS[0]="chain"

CKPTS[1]="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed2000/last.pt"
OUTDIRS[1]="runs/report8/stagewise/n40_pathunion_seed2000"
SPLITS[1]="4 7 8 10 14 20"; TOPOS[1]="chain"

CKPTS[2]="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed3000/last.pt"
OUTDIRS[2]="runs/report8/stagewise/n40_pathunion_seed3000"
SPLITS[2]="4 7 8 10 14 20"; TOPOS[2]="chain"

CKPTS[3]="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed4000/last.pt"
OUTDIRS[3]="runs/report8/stagewise/n40_pathunion_seed4000"
SPLITS[3]="4 7 8 10 14 20"; TOPOS[3]="chain"

CKPTS[4]="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed1000/last.pt"
OUTDIRS[4]="runs/report8/stagewise/n40_pathunion_cycle_seed1000"
SPLITS[4]="4 20"; TOPOS[4]="cycle"

CKPTS[5]="runs/report9/n46_train/n46_path_union_roberta_similarity_lam0_seed1000/last.pt"
OUTDIRS[5]="runs/report9/stagewise/n46_pathunion_seed1000"
SPLITS[5]="4 23"; TOPOS[5]="chain"

I=$SLURM_ARRAY_TASK_ID
CKPT="${CKPTS[$I]}"; OUTDIR="${OUTDIRS[$I]}"; SPLITS_I="${SPLITS[$I]}"; TOPO="${TOPOS[$I]}"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi
echo "task $I: checkpoint=$CKPT output=$OUTDIR splits=($SPLITS_I) topology=$TOPO"

python stagewise_diagnostics.py \
  --checkpoint "$CKPT" \
  --output_dir "$OUTDIR" \
  --splits $SPLITS_I \
  --topology "$TOPO" \
  --n_graphs 64

echo "done task $I"
