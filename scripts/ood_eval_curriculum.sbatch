#!/bin/bash
#SBATCH --job-name="ood_curriculum"
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
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
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "================"

# IMPORTANT: edit CKPT below to point at the curriculum run you want to evaluate.
# Easiest: list the runs/ directory after the training job finishes and copy the
# path of the n40_p005_curriculum/best.pt that was produced.
CKPT="${CURRICULUM_CKPT:-$(ls -1dt runs/curriculum_er_n40_*/n40_p005_curriculum/best.pt 2>/dev/null | head -n 1)}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: could not find curriculum checkpoint."
    echo "       Set CURRICULUM_CKPT=path/to/best.pt before submitting,"
    echo "       or make sure the curriculum training job has finished."
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"

OUT_ROOT="runs/ood_eval_curriculum_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

python experiments2/ood_evaluation.py \
    --mode n40_to_structured \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_ROOT/er_n40_curriculum_to_structured" \
    --num_workers 16

echo ""
echo "Done. Output in $OUT_ROOT"
