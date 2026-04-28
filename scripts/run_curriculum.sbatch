#!/bin/bash
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=graph-curr
#SBATCH --output=logs/graph-curriculum-%j.out
#SBATCH --error=logs/graph-curriculum-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it

source ~/.bashrc
conda activate graph_tf
cd ~/transformer-for-graphs

mkdir -p logs
export PYTHONUNBUFFERED=1

which python
python --version
python -c "import torch; print('torch.__version__=', torch.__version__); print('torch.version.cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('cuda_device_count=', torch.cuda.device_count())"
date
nvidia-smi || true

python -u experiments/curriculum_diameter_dynamics.py \
  --device cuda \
  --p 0.08 \
  --n 20 \
  --batch_size 128 \
  --total_steps 5000 \
  --eval_every_steps 200 \
  --t1 2000 \
  --eval_size 1000 \
  --d_model 128 \
  --d_ff 256 \
  --n_layers 2 \
  --n_heads 4 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --max_attempts 200000 \
  --seed 42
