#!/bin/bash
#SBATCH --job-name="embed_geom"
#SBATCH --account=3352759
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00

mkdir -p out err
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
echo "Node: $(hostname)"

# Experiment A: embedding geometry of trained checkpoints.
# n=40 connectivity model (L=2) on the 2chain wall case and on ER:
UNF=runs/retrain_er_n40_big_495903/n40_p005_unfiltered_big/best.pt
python analyze_embeddings.py --checkpoint $UNF --kind twochains
python analyze_embeddings.py --checkpoint $UNF --kind er

# depth-sweep reach models on path-union (does the separation extend with L?):
for L in 1 2 3 4; do
  CK=$(ls -d runs/reach_depth/reach_n64_L${L}_* 2>/dev/null | head -1)/last.pt
  [ -f "$CK" ] && python analyze_embeddings.py --checkpoint "$CK" --kind pathunion
done
