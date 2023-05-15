#!/usr/bin/env bash


module load miniconda

source activate rnp

srun -c 4 --time=05:00:00 --mem 10G python bayesopt/bo_apply.py
