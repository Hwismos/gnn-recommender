#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx

#SBATCH --job-name=LGCN.out
#SBATCH -o SLURM.%N.LGCN.out
#SBATCH -e SLURM.%N.LGCN.err

#SBATCH --gres=gpu:hgx

hostname
date

module add ANACONDA/2020.11
module add CUDA/11.2.2
CUDA_VISIBLE_DEVICES=2 python -u ./code/main.py > LGCN.out