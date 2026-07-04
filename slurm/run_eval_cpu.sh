#!/bin/bash --login
#SBATCH -p interactive          # CPU fallback when gpuL sits at N/A (Phase-6-proven: schedules now)
#SBATCH -n 8
#SBATCH -t 0-05:00:00           # under the interactive 6 h cap
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# CPU fallback for the dense eval when the GPU queue is blocked by the teaching account's low
# fair-share. Runs the SAME code on CPU (torch auto-detects no GPU); float32 rankings match the
# GPU run. Premise embeddings are cached to the index dir on first build, so if this times out you
# can resubmit and it reloads them, only re-encoding states. See docs/CLUSTER.md and phase7.md.
# dos2unix this file before sbatch. Usage: sbatch slurm/run_eval_cpu.sh configs/dense_reprover.yaml
module purge
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
