#!/bin/bash
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it
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
python -c "import torch; print('torch.__version__=', torch.__version__); print('torch.version.cuda=', getattr(torch.version, 'cuda', None)); print('cuda_available=', torch.cuda.is_available()); print('cuda_device_count=', torch.cuda.device_count())"

python experiments/restrict_diameter_dynamics.py \
  --p 0.2 \
  --train_size 50000 \
  --epochs 50 \
  --batch_size 128 \
  --eval_every_steps 200 \
  --eval_size 1000
  --device cuda