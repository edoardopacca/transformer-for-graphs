#!/bin/bash
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=graph-rd-dyn
#SBATCH --output=logs/graph-rd-dyn-%j.out
#SBATCH --error=logs/graph-rd-dyn-%j.err

source ~/.bashrc
conda activate graph_tf

cd ~/transformer-for-graphs

mkdir -p logs
mkdir -p trainings
mkdir -p runs

which python
python --version
nvidia-smi

python experiments/restrict_diameter_dynamics.py \
  --p 0.2 \
  --train_size 50000 \
  --epochs 50 \
  --batch_size 128 \
  --eval_every_steps 200 \
  --eval_size 1000