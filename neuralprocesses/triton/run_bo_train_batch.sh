#!/usr/bin/env bash
#SBATCH --mem=10GB
#SBATCH --cpus-per-task 4
#SBATCH --time=05:00:00
#SBATCH --array=1-3
#SBATCH --gres=gpu:1


module load miniconda

source activate rnp

srun  python bayesopt/bo_train.py --target hartmann3d --save_postfix "_$SLURM_ARRAY_TASK_ID" \
        --run "$SLURM_ARRAY_TASK_ID" --model "cnp"
