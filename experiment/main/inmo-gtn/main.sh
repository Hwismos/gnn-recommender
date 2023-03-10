#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx

#SBATCH --job-name=job-mv.out
#SBATCH -o ./result/movie-len/out/SLURM.%N.%j.out
#SBATCH -e ./result/movie-len/err/SLURM.%N.%j.err

#SBATCH --gres=gpu:hgx

hostname
date

# module add ANACONDA/2020.11
# module add CUDA/11.2.2

# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./inmo_code/hyperparameter/igcn_tuning.py > ./result/movie-len/res/IGCN-result.out
CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./inmo_code/hyperparameter/lgcn_tuning.py > ./result/movie-len/res/LightGCN-result.out
# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./gtn_code/run_main.py > ./result/movie-len/res/GTN-embedding-check.out
