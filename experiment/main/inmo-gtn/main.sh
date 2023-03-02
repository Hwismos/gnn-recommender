#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx

#SBATCH --job-name=test.out
#SBATCH -o ./result/SLURM.%N.test.out
#SBATCH -e ./result/SLURM.%N.test.err

#SBATCH --gres=gpu:hgx

hostname
date

# module add ANACONDA/2020.11
# module add CUDA/11.2.2

# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./inmo_code/hyperparameter/igcn_tuning.py > ./result/I-GTN_using_se_loss_gowalla_version1.out
# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./gtn_code/run_main.py > ./result/I-GTN_using_se_loss_gowalla_version1.out
# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./gtn_code/run_main.py > ./result/I-GTN_using_se_loss_gowalla_version2.out
# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./inmo_code/hyperparameter/igcn_tuning.py > ./result/I-GCN_code_analysis.out
# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./gtn_code/run_main.py > ./result/I-GTN_using_se_loss_gowalla_version3.out

CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./gtn_code/run_main.py > ./result/test.out
# CUDA_VISIBLE_DEVICES=0 /home1/prof/hwang1/miniconda3/envs/seokhwi2/bin/python3.8 -u ./inmo_code/hyperparameter/igcn_tuning.py > ./result/test.out
