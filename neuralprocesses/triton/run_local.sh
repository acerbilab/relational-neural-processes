#!/usr/bin/env bash


# module load miniconda

# source activate rnp

#srun -c 4 --time=00:15:00 --gres=gpu:1 python tmp.py
python hyper.py
# srun -c 4 --time=00:15:00  python hyper.py
