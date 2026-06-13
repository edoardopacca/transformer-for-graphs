#!/bin/bash
#SBATCH --job-name="ood_n14"
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

OUT_ROOT="runs/report2/ood_eval_n14_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

echo ""
echo "###################################################"
echo "# 2chains (n=14) -> ER(n=14, p in {0.05, 0.2, 0.5})"
echo "###################################################"
python experiments2/ood_evaluation.py \
    --mode n14_to_er \
    --checkpoint runs/report2/retrain_2chains_3352759/n14_k7_rep0/best.pt \
    --output_dir "$OUT_ROOT/2chains_to_er" \
    --num_workers 16

echo ""
echo "###################################################"
echo "# 2cliques (n=14) -> ER(n=14, p in {0.05, 0.2, 0.5})"
echo "###################################################"
python experiments2/ood_evaluation.py \
    --mode n14_to_er \
    --checkpoint runs/report2/retrain_2cliques_3352759/n14_k7_rep0/best.pt \
    --output_dir "$OUT_ROOT/2cliques_to_er" \
    --num_workers 16

echo ""
echo "Done. Output in $OUT_ROOT"
