#!/bin/bash
#SBATCH --job-name="ood_n40"
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

OUT_ROOT="runs/report2/ood_eval_n40_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

# The four ER n=40 checkpoints to evaluate. Update the paths if needed.
declare -A CHECKPOINTS=(
    [unfiltered]="runs/report2/retrain_er_n40_485350/n40_p005_unfiltered/best.pt"
    [diam11]="runs/report2/retrain_er_n40_485352/n40_p005_diam11/best.pt"
    [diam9]="runs/report2/retrain_er_n40_485351/n40_p005_diam9/best.pt"
    [diam7]="runs/report2/retrain_er_n40_485353/n40_p005_diam7/best.pt"
)

for tag in unfiltered diam11 diam9 diam7; do
    ckpt="${CHECKPOINTS[$tag]}"
    echo ""
    echo "###################################################"
    echo "# ER n=40 / $tag -> 2chains + 2cliques (n_active variable)"
    echo "# checkpoint: $ckpt"
    echo "###################################################"
    python experiments2/ood_evaluation.py \
        --mode n40_to_structured \
        --checkpoint "$ckpt" \
        --output_dir "$OUT_ROOT/er_n40_${tag}_to_structured" \
        --num_workers 16
done

echo ""
echo "Done. Output in $OUT_ROOT"
