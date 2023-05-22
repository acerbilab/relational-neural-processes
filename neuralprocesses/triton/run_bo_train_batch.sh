#!/usr/bin/env bash
#SBATCH --mem=10GB
#SBATCH --cpus-per-task 4
#SBATCH --time=05:00:00
#SBATCH --array=0-5
#SBATCH --gres=gpu:1

files=("cnp" "gnp" "rcnp" "rgnp"  "acnp" "agnp")
RUN=1

module load miniconda

source activate rnp

srun  python bayesopt/bo_train.py --target hartmann3d \
        --exp "bo_fixed" \
        --run "$RUN" \
        --model "${files[$SLURM_ARRAY_TASK_ID]}"
