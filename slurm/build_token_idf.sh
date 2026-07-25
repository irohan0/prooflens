#!/bin/bash --login
#SBATCH -p multicore            # CPU only — tokenization, no GPU needed
#SBATCH -n 8
#SBATCH -t 0-01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Phase 22 -- build the corpus token->IDF table (one-time, split-agnostic).
# dos2unix this file before sbatch. Usage: sbatch slurm/build_token_idf.sh
module purge
module load python/3.13.1
cd ~/scratch/prooflens
source .venv/bin/activate
export PYTHONPATH=$PWD/src
export SCRATCH=$HOME/scratch
export DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4
export MODELS_DIR=$HOME/scratch/prooflens_data/models
export OMP_NUM_THREADS=$SLURM_NTASKS
mkdir -p "$SCRATCH/prooflens/indices"
# The base ColBERT is only a tokenizer source (fine-tuning doesn't change the vocabulary), so ONE
# table serves both the random and novel fine-tuned indices.
python scripts/build_token_idf.py \
  --model "$MODELS_DIR/lightonai__GTE-ModernColBERT-v1" \
  --out "$SCRATCH/prooflens/indices/token_idf.json"
