#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx

#SBATCH --job-name=TEST.out
#SBATCH -o SLURM.%N.TEST.out
#SBATCH -e SLURM.%N.TEST.err

#SBATCH --gres=gpu:hgx

hostname
date

# module add ANACONDA/2020.11
# module add CUDA/11.2.2
CUDA_VISIBLE_DEVICES=2 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./run/run.py > ./test.out