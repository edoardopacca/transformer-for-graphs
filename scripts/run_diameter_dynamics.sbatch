#!/bin/bash
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=graph-rd-sweep
#SBATCH --output=logs/graph-rd-sweep-%j.out
#SBATCH --error=logs/graph-rd-sweep-%j.err

source ~/.bashrc
conda activate graph_tf

cd ~/transformer-for-graphs

mkdir -p logs
mkdir -p trainings
mkdir -p runs

which python
python --version
nvidia-smi

python experiments/restrict_diameter_sweep.py \
  --p_min 0.05 \
  --p_max 0.55 \
  --p_steps 11 \
  --train_size 50000 \
  --val_size 2000 \
  --epochs 50 \
  --batch_size 128