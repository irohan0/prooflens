#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G               # multicore cap is 8192 MB/core; 24G/4 = 6144 MB/core (BM25 needs ~4G)
#SBATCH -t 0-06:00:00
#SBATCH --job-name=bm25_eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# BM25 evaluation on CSF3 multicore (CPU). One split per job.
# Usage: sbatch slurm/bm25_eval.sh random   |   sbatch slurm/bm25_eval.sh novel_premises
module purge
module load python/3.13.1
cd ~/scratch/prooflens
source .venv/bin/activate
export PYTHONPATH=$PWD/src
export SCRATCH=$HOME/scratch
export DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Optional 2nd arg = subset size (first N examples) -> quick backfillable diagnostic run,
# written to results_subset/ so it never overwrites the full-run metrics.
if [ -n "$2" ]; then
  python scripts/run_eval.py --config configs/bm25.yaml --split "$1" --limit "$2" --results-dir results_subset
else
  python scripts/run_eval.py --config configs/bm25.yaml --split "$1"
fi
