#!/bin/bash
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=graph-rd-pair
#SBATCH --output=logs/graph-rd-pairwise-%j.out
#SBATCH --error=logs/graph-rd-pairwise-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it

source ~/.bashrc
conda activate graph_tf
cd ~/transformer-for-graphs

mkdir -p logs
export PYTHONUNBUFFERED=1

which python
python --version
date

python -u experiments/restrict_diameter_dynamics_pairwise.py --p 0.2

date