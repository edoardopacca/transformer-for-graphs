#!/bin/bash
#SBATCH --job-name="ood_curriculum_big"
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=01:00:00

mkdir -p out err

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

echo "=== Job info ==="
echo "Node:  $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "================"

# Auto-detect the most recent big curriculum checkpoint
CKPT="${CURRICULUM_BIG_CKPT:-$(ls -1dt runs/curriculum_er_n40_big_*/n40_p005_curriculum_big/best.pt 2>/dev/null | head -n 1)}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no big curriculum checkpoint found at runs/curriculum_er_n40_big_*/n40_p005_curriculum_big/best.pt"
    echo "Set CURRICULUM_BIG_CKPT=path/to/best.pt manually, or wait for the training job to finish."
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"

OUT_ROOT="runs/ood_eval_curriculum_big_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

python experiments2/ood_evaluation.py \
    --mode n40_to_structured \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_ROOT/er_n40_big_curriculum_to_structured" \
    --num_workers 16

echo ""
echo "Done. Output in $OUT_ROOT"
