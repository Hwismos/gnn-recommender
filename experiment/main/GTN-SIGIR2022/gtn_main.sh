#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx

#SBATCH --job-name=GTN.out
#SBATCH -o SLURM.%N.GTN.out
#SBATCH -e SLURM.%N.GTN.err

#SBATCH --gres=gpu:hgx

hostname
date

# module add ANACONDA/2020.11
# module add CUDA/11.2.2
CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./code/run_main.py > ./test/GTN_amazon_1000epochs.out
# CUDA_VISIBLE_DEVICES=2 python -u ./code/run_main.py > ./test/GTN_1000epochs.out
