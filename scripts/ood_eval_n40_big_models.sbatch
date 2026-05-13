#!/bin/bash
#SBATCH --job-name="ood_n40_big"
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=02:00:00

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
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "================"

OUT_ROOT="runs/ood_eval_n40_big_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

# Auto-detect the most recent big checkpoint for each variant.
# The training script writes:  runs/retrain_er_n40_big_<JOBID>/n40_p005_<tag>_big/best.pt
for tag in unfiltered diam11 diam9 diam7; do
    case "$tag" in
        unfiltered) subdir="n40_p005_unfiltered_big" ;;
        diam7)      subdir="n40_p005_diam7_big" ;;
        diam9)      subdir="n40_p005_diam9_big" ;;
        diam11)     subdir="n40_p005_diam11_big" ;;
    esac

    # Most recent JOBID directory containing this variant
    ckpt=$(ls -1dt runs/retrain_er_n40_big_*/${subdir}/best.pt 2>/dev/null | head -n 1)
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo ""
        echo "!!! SKIP $tag: no checkpoint found at runs/retrain_er_n40_big_*/${subdir}/best.pt"
        continue
    fi
    echo ""
    echo "###################################################"
    echo "# Big ER n=40 / $tag  ->  OOD (2chains + 2cliques + unfilt ER)"
    echo "# checkpoint: $ckpt"
    echo "###################################################"
    python experiments2/ood_evaluation.py \
        --mode n40_to_structured \
        --checkpoint "$ckpt" \
        --output_dir "$OUT_ROOT/er_n40_big_${tag}_to_structured" \
        --num_workers 16
done

echo ""
echo "Done. Output in $OUT_ROOT"
