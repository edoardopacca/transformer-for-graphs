#!/bin/bash
#SBATCH --job-name="sweep_rob_unrestr"
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-4

mkdir -p out err
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate graph_tf

# seed sweep: 4000, 5000, 6000, 7000, 8000 (new seeds, no collision with 1000-3000)
SEED=$(( 4000 + SLURM_ARRAY_TASK_ID * 1000 ))

echo "Node: $(hostname) | array ${SLURM_ARRAY_TASK_ID} -> seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments2/reproduce_paper_n20.py \
    --output_root "runs/report3/repro_paper_n20_roberta" \
    --arch roberta \
    --num_workers 16 \
    --train_steps 1000000 \
    --batch_size 1000 \
    --seed ${SEED}
