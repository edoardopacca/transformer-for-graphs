#!/bin/bash
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=graph-capacity
#SBATCH --output=logs/graph-capacity-%j.out
#SBATCH --error=logs/graph-capacity-%j.err

source ~/.bashrc
conda activate graph_tf

cd ~/transformer-for-graphs

mkdir -p logs
mkdir -p trainings
mkdir -p runs

which python
python --version
nvidia-smi

python experiments/capacity_test.py
