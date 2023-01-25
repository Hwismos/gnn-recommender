# parse 모듈의 parse_args 메소드 임포트
# 어쩌피 메소드 하나 밖에 없음
from parse import parse_args        
import os
from os.path import join    # str을 합쳐서 경로로 만들어주는 메소드
import torch
from enum import Enum   # Enumerate
import multiprocessing

# 여기서도 난수 생성
seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# 클래스나 메소드 없음
# 순차적으로 실행되는 단순한 모듈

# 딕셔너리 타입
# 환경변수의 특정 값을 True로 바꾸나봄
os.environ['KMP_DUPLICATE_LIB_OK']='True'

args=parse_args()   # 파싱한 인자를 받아옴

# str의 join 메소드를 이용해서 경로 변수 설정
ROOT_PATH='./'  # 현재 위치 설정
CODE_PATH=join(ROOT_PATH, 'code')
DATA_PATH=join(ROOT_PATH, 'data')
BOARD_PATH=join(ROOT_PATH, 'runs')
FILE_PATH=join(ROOT_PATH, 'checkpoints')

import sys

# 인터프리터가 참조하는 경로 추가
sys.path.append(join(CODE_PATH, 'sources'))

# checkpoints 디렉토리 없으면 만들고 만들면 다시 만들지 않음
if not os.path.exits(FILE_PATH):
    os.makedirs(FILE_PATH, exits_ok=True)

# 파싱한 인자로 world.config 딕셔너리 초기화
# world.config로 계속 이용
config={}
all_dataset = ['gowalla', 'yelp2018', 'amazon-book', 'last-fm']
all_models = ['mf', 'gtn', 'lgn']
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
config['dataset'] = args.dataset
config['epochs'] = args.epochs
config['lambda2'] = args.lambda2

# gpu 설정
GPU = torch.cuda.is_available()
torch.cuda.set_device(args.gpu_id)
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

# 데이터셋과 모델이름 저장
dataset=args.dataset
model_name=args.model

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path        # weight 파일을 저장할 checkpoints 경로 저장
topks = eval(args.topks)    # str을 숫자로 변환
tensorboard = args.tensorboard
comment = args.comment

from warnings import simplefilter

# 뭔지 모르겠음
# 경고해주는 것 같음
simplefilter(action='ignore', category=FutureWarning)   

# 특수 형태의 print
def cprint(words: str):
    print("##########################")
    print(f"\033[0;30;43m{words}\033[0m")