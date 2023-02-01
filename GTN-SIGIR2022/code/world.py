'''
## GTN-pytorch
"Graph Trend Filtering Networks for Recommendations", Accepted by SIGIR'2022.
Pytorch Implementation of GTN in Graph Trend Filtering Networks for Recommendations
The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch

@inproceedings{fan2022graph,
  title={Graph Trend Filtering Networks for Recommendations},
  author={Fan, Wenqi and Liu, Xiaorui and Jin, Wei and Zhao, Xiangyu and Tang, Jiliang and Li, Qing},
  booktitle={International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2022}
}
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 입력값 파싱
# parse 모듈의 parse_args 메소드 이용
# parse 모듈 import 끝
args = parse_args()

ROOT_PATH = "/home/hwiric/Internship/GTN-SIGIR2022"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'run_GTN')
FILE_PATH = join(CODE_PATH, 'checkpoints_GTN')

# print(os.getcwd())  →   /home/hwiric/Internship/GTN-SIGIR2022/code
# print(f'BOARD_PATH: {BOARD_PATH}\nFILE_PATH: {FILE_PATH}')

import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


# 모든 데이터셋과 모델들을 리스트로 정의
all_dataset = ['gowalla', 'yelp2018', 'amazon-book', 'lastfm']
all_models = ['mf', 'gtn', 'lgn']

# config(환경설정) 딕셔너리를 정의
# --key format으로 value에 접근할 수 있음
config = {}
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['K'] = args.K
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['args'] = args
# python run_main.py --dataset 'gowalla'
config['dataset'] = args.dataset
config['epochs'] = args.epochs
config['lambda2'] = args.lambda2

GPU = torch.cuda.is_available()
torch.cuda.set_device(args.gpu_id)
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
# CORES = multiprocessing.cpu_count()
seed = args.seed

dataset = args.dataset
model_name = args.model

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
# eval 메소드: 매개변수로 받은 expression(식)을 문자열로 받아서 실행하는 함수
# eval("1+2") -> 3
topks = eval(args.topks)
# 텐서보드: 텐서블로우의 시각화 도구
tensorboard = args.tensorboard
comment = args.comment

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words: str):
    print("##########################")
    print(f"\033[0;30;43m{words}\033[0m")
