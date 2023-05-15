#!/usr/bin/env bash


module load miniconda

source activate rnp

srun -c 4 --time=01:00:00 --gres=gpu:1 python bo_apply.py
