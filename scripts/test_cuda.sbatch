#!/bin/bash
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=test-cuda
#SBATCH --output=logs/test-cuda-%j.out
#SBATCH --error=logs/test-cuda-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3352759@studbocconi.it

source ~/.bashrc
conda activate graph_tf

cd ~/transformer-for-graphs
mkdir -p logs

which python
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda_version', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
nvidia-smi || true
