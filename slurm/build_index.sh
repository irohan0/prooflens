#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 32
#SBATCH -t 0-08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# CPU index / BM25 build (AMD Genoa multicore). See docs/CLUSTER.md §4.
# dos2unix this file before sbatch. Usage: sbatch slurm/build_index.sh configs/<cfg>.yaml
module purge
module load python/3.13.1
cd ~/scratch/prooflens
source .venv/bin/activate
export PYTHONPATH=$PWD/src
export SCRATCH=$HOME/scratch
export DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4
export MODELS_DIR=$HOME/scratch/prooflens_data/models
export OMP_NUM_THREADS=$SLURM_NTASKS
python scripts/build_index.py --config "$1"
