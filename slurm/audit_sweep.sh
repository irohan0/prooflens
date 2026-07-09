#!/bin/bash --login
#SBATCH -p gpuL                 # L40S partition on CSF3 (GRES gpu:l40s:4)
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 8                    # CPU cores to feed the GPU + the numpy MaxSim scoring
#SBATCH -t 0-04:00:00           # tight -> backfills sooner
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Phase 11 pre-fine-tuning sweep (symbol-weight + query-length), reusing the cached Phase-8 LI
# premise index (eval-only, no re-indexing). Prints a variant x {R@1,R@10,MRR,nDCG@10} table to
# the .out. dos2unix this file before sbatch.
# Usage: sbatch slurm/audit_sweep.sh [config] [split] [limit]
#   e.g. sbatch slurm/audit_sweep.sh configs/late_interaction.yaml random 500
module purge
module load cuda/12.6.2
module load python/3.13.1
cd ~/scratch/prooflens
source .venv/bin/activate
export PYTHONPATH=$PWD/src
export SCRATCH=$HOME/scratch
export DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4
export MODELS_DIR=$HOME/scratch/prooflens_data/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_NTASKS
python scripts/audit.py --part sweep \
  --config "${1:-configs/late_interaction.yaml}" \
  --split "${2:-random}" \
  --limit "${3:-500}"
