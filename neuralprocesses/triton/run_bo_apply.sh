#!/usr/bin/env bash
#SBATCH --mem=10GB
#SBATCH --cpus-per-task 8
#SBATCH --time=02:00:00
#SBATCH --array=1-3


module load miniconda

source activate rnp

srun python bayesopt/bo_apply.py \
  --exp "bo_fixed" \
  --target_name "hartmann3d" \
  --vis \
  --n_rep 10 \
  --run "$SLURM_ARRAY_TASK_ID" \
  --n_steps 50

