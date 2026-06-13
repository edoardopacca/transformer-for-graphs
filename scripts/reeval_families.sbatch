#!/bin/bash
#SBATCH --job-name=reeval_fam
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Re-run eval_families.py (updated: disc/reach split + barbell_var + expander_var)
# on every existing checkpoint in one job. Overwrites the auto-eval produced by
# older code. Light (a couple of minutes per checkpoint).
#
#   sbatch scripts/reeval_families.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

# new families-experiment runs (last.pt)
for d in runs/report4/families_n20/*/ ; do
  [ -f "$d/last.pt" ] && echo "== $d ==" && \
    python eval_families.py --checkpoint "$d/last.pt" --output_dir "$d/families" --size 4000
done
# Report-III roberta baselines (best.pt)
for d in runs/report3/repro_paper_n20_roberta/n20_p008_unrestricted_seed*/ ; do
  [ -f "$d/best.pt" ] && echo "== $d ==" && \
    python eval_families.py --checkpoint "$d/best.pt" --output_dir "$d/families" --size 4000
done
echo "re-eval done"
