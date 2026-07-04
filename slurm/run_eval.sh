#!/bin/bash --login
#SBATCH -p gpuL                 # L40S partition on CSF3 (confirmed via sinfo; GRES gpu:l40s:4)
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 8                    # CPU cores to feed the GPU
#SBATCH -t 0-04:00:00           # tight -> backfills sooner (GPU max 4-0)
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# GPU encode + evaluate (L40S). Phases 7-9. See docs/CLUSTER.md §4.
# dos2unix this file before sbatch. Usage: sbatch slurm/run_eval.sh configs/<cfg>.yaml
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
python scripts/run_eval.py --config "$1"
