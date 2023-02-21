#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx

#SBATCH --job-name=I-GTN_gowall_1000epochs.out
#SBATCH -o ./result/SLURM.%N.I-GTN_gowall_1000epochs.out
#SBATCH -e ./result/SLURM.%N.I-GTN_gowall_1000epochs.err

#SBATCH --gres=gpu:hgx

hostname
date

# module add ANACONDA/2020.11
# module add CUDA/11.2.2

# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./inmo_code/hyperparameter/igcn_tuning.py > ./result/IGCN_gowall_1000epochs.out
CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./gtn_code/run_main.py > ./result/I-GTN_gowall_1000epochs.out


