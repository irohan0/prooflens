#!/bin/bash --login
#SBATCH -p gpuL                 # L40S partition on CSF3 (GRES gpu:l40s:4)
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 8                    # CPU cores to feed the GPU
#SBATCH -t 1-00:00:00           # same budget class as the LI training job
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Fine-tune the MATCHED single-vector control (Phase 15). Identical environment to slurm/train.sh
# (the LI training job) -- only the entrypoint differs -- so the control's budget/env match the LI
# run exactly. dos2unix before sbatch.
# Usage: sbatch slurm/train_sv.sh configs/train/sv_ft_random.yaml [--limit N]
#   e.g. sbatch slurm/train_sv.sh configs/train/sv_ft_random.yaml --limit 2000   # quick trial
#        sbatch slurm/train_sv.sh configs/train/sv_ft_random.yaml               # full run
#        sbatch slurm/train_sv.sh configs/train/sv_ft_novel.yaml                # full run (novel)
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
python scripts/train_sv.py --config "$@"
