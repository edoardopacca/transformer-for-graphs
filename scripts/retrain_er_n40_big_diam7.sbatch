#!/bin/bash
#SBATCH --job-name="big_er_n40_diam7_exp2"
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=23:55:00

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
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| bf16:', torch.cuda.is_bf16_supported())"
echo "================"

# NOTE: D≤7 has 14% acceptance rate on ER(40, 0.05). With 1B samples needed,
# rejection sampling generates ~7B candidate graphs. Online data generation
# might become a bottleneck with only 16 CPUs — if you see GPU stalls in the
# log, the fix is to increase --cpus-per-task above (32 or 60 if the partition
# allows it on the H200 nodes).
python experiments2/retrain_and_test_er_n40_big.py \
    --output_root "runs/retrain_er_n40_big_exp2_${SLURM_JOB_ID}" \
    --max_diameter 7 \
    --num_workers 32 \
    --train_steps 1000000 \
    --batch_size 1000
