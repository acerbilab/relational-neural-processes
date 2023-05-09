#!/usr/bin/env bash


module load miniconda

source activate rnp

srun -c 4 --time=05:00:00 --gres=gpu:1 python bo_train.py
