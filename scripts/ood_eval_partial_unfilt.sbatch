#!/bin/bash
#SBATCH --job-name="ood_partial_unfilt"
#SBATCH --account=3352759
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:15:00

mkdir -p out err

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

# Auto-trova il checkpoint last.pt più recente dell'unfiltered big training
# (mentre il training continua a riscriverlo ogni 5k step).
CKPT=$(ls -1dt runs/report2/retrain_er_n40_big_*/n40_p005_unfiltered_big/last.pt 2>/dev/null | head -n 1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no unfiltered big checkpoint found"
    exit 1
fi

# Inferisce a quale step siamo (lo script salva 'step' nel checkpoint)
STEP=$(python -c "import torch; ck=torch.load('$CKPT', map_location='cpu', weights_only=False); print(ck.get('step', '?'))" 2>/dev/null)

echo "=== Partial OOD eval ==="
echo "Checkpoint: $CKPT"
echo "Step:       $STEP"
echo "Node:       $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "========================"

OUT_DIR="runs/report2/ood_partial_unfilt_step${STEP}_${SLURM_JOB_ID}"

python experiments2/ood_evaluation.py \
    --mode n40_to_structured \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_DIR" \
    --num_workers 16

echo ""
echo "Done. Output in $OUT_DIR"
