#!/bin/bash --login
#SBATCH -p gpuL                 # L40S partition on CSF3 (GRES gpu:l40s:4)
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 8                    # CPU cores to feed the GPU
#SBATCH -t 1-00:00:00           # fine-tuning is longer than eval; still tight for backfill
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Fine-tune the late-interaction (ColBERT) retriever (Phase 13+). dos2unix before sbatch.
# Usage: sbatch slurm/train.sh configs/train/li_ft_random.yaml [--limit N]
#   e.g. sbatch slurm/train.sh configs/train/li_ft_random.yaml            # full run
#        sbatch slurm/train.sh configs/train/li_ft_random.yaml --limit 2000   # quick trial
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
python scripts/train_li.py --config "$@"
