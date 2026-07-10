#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 16
#SBATCH -t 0-06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Build fine-tuning triplets (train + val) for one split — CPU (bm25s hard-negative mining).
# dos2unix before sbatch. Needs: pip install bm25s.
# If it sits PENDING on multicore (teaching-account fair-share, cf. Phase 6), switch
#   -p multicore  ->  -p interactive   (6 h cap; scheduled instantly in Phase 6; a build is ~1 h).
# Usage: sbatch slurm/build_pairs.sh <split> <negatives> [n_neg] [cap]
#   e.g. sbatch slurm/build_pairs.sh random bm25 3 300
module purge
module load python/3.13.1
cd ~/scratch/prooflens
source .venv/bin/activate
export PYTHONPATH=$PWD/src
export SCRATCH=$HOME/scratch
export DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4
export OMP_NUM_THREADS=$SLURM_NTASKS
SPLIT="$1"; NEG="$2"; NNEG="${3:-3}"; CAP="${4:-300}"
echo "== build TRAIN triplets: split=$SPLIT neg=$NEG n_neg=$NNEG cap=$CAP =="
python scripts/build_pairs.py --config configs/late_interaction.yaml \
  --split "$SPLIT" --split-file train.json --negatives "$NEG" --n-neg "$NNEG" --cap "$CAP"
echo "== build VAL triplets (for checkpoint selection) =="
python scripts/build_pairs.py --config configs/late_interaction.yaml \
  --split "$SPLIT" --split-file val.json --negatives "$NEG" --n-neg "$NNEG" --cap "$CAP"
echo "== done =="
