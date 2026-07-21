#!/bin/bash --login
#SBATCH -p gpuL                 # L40S partition on CSF3 (GRES gpu:l40s:4)
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 8                    # CPU cores to feed the GPU
#SBATCH -t 0-08:00:00           # byte-level ByT5 over a ~2300-byte context is not cheap
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Phase 21 -- retrieval-augmented tactic generation (offline; no Lean, no Dojo).
# dos2unix this file before sbatch.
#
#   sbatch slurm/generate_eval.sh configs/generate/gen_ft_li_novel.yaml
#   sbatch slurm/generate_eval.sh configs/generate/gen_ft_li_novel.yaml --limit 50   # pilot
#
# Conditions are INDEPENDENT jobs -- submit them in parallel. Run a --limit pilot first and read
# `seconds_per_example` out of the results JSON before committing to a full split.
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
CONFIG="$1"; shift
python scripts/run_generate.py --config "$CONFIG" "$@"
