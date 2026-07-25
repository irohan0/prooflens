#!/bin/bash
# Phase 25 — submit the multi-seed matched-control sweep with Slurm dependencies, fire-and-forget.
#
# For each split (random, novel) and seed, chains via --dependency=afterok:
#   LI:  train -> eval OFF (builds the ~3h index) -> eval IDF (reuses that index, fast)
#   SV:  train -> eval
# so nothing runs before its checkpoint exists and no manual waiting is needed.
#
# Prereqs (once):
#   - configs/seeds/ generated:  python scripts/make_seed_configs.py --seeds 1 2 3 4
#   - jobscripts unix'd:         dos2unix slurm/*.sh
#   - token_idf.json built (Phase 22) and the base *_bm25_{train,val}.jsonl pairs staged.
# Usage:  bash slurm/submit_seed_sweep.sh "1 2 3 4"     (default seeds: 1 2 3 4)
set -euo pipefail
SEEDS="${1:-1 2 3 4}"
C=configs/seeds

for split in random novel; do
  for s in $SEEDS; do
    li_tr=$(sbatch --parsable slurm/train.sh "$C/li_ft_${split}_s${s}.yaml")
    li_off=$(sbatch --parsable --dependency=afterok:"$li_tr" \
             slurm/run_eval.sh "$C/late_interaction_ft_${split}_s${s}.yaml")
    li_idf=$(sbatch --parsable --dependency=afterok:"$li_off" \
             slurm/run_eval.sh "$C/late_interaction_ft_${split}_idf_s${s}.yaml")
    sv_tr=$(sbatch --parsable slurm/train_sv.sh "$C/sv_ft_${split}_lr3e6_s${s}.yaml")
    sv_ev=$(sbatch --parsable --dependency=afterok:"$sv_tr" \
            slurm/run_eval.sh "$C/dense_sv_ft_${split}_lr3e6_s${s}.yaml")
    echo "[$split s$s] LI: train=$li_tr off=$li_off idf=$li_idf | SV: train=$sv_tr eval=$sv_ev"
  done
done
echo "submitted. watch with: squeue -u \$USER  (evals show Dependency until their train finishes)"
