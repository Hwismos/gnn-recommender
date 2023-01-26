import world
import utils
import dataloader
from utils import timer
import model

from pprint import pprint
from time import time

import numpy as np
import torch

# auc는 roc 커브의 면적으로, 1에 가까울수록 좋은 모델이라 평가 받음
from sklearn.metrics import roc_auc_score   

# 진행 상황을 보여주는 기능을 제공하는 라이브러리
from tqdm import tqdm       
import multiprocessing

import random
import numpy as np

seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

CORES = multiprocessing.cpu_count() // 4

# nn.Module에서 train time과 eval time에서 수행하는 작업을 분리
# eval() 메소드는 evaluation 과정에서 사용하지 않아야 할, Dropout layer, 레이어들을 off 시킴
# ! getEmbedding, getUsersRating 메소드에 주목하는 이유는, 두 메소드가 모두 computer 메소드를 호출하기 때문
    # computer 메소드가 실질적으로 임베딩을 학습시킴

# ===================================TRAIN=======================================

# 훈련은 forward가 알아서 함
# 각 훈련에 대한 평균 loss를 계산해서 반환하는 역할을 함
# torch.nn.Module.train()
# ! model.bpr_loss 메소드는 getEmbedding 메소드를 참조
    # 학습을 위해 사용됨
    # bpr_loss 메소드는 utils의 BPRLoss 객체에서 사용됨
    # BPRLoss 객체는 아래 메소드의 인자로 전달됨
def BPR_train_original(dataset,
                        recommend_model: model.GTN,
                        loss_clas: utils.BPRLoss,
                        epoch,
                        neg_k=1,
                        w=None) -> str:
    pass

# ===============================================================================

# ===================================TEST========================================

def test_one_batch(X) -> dict:
    pass

# torch.nn.Module.eval()
# ! model.getUsersRating 메소드 참조
def Test(dataset,
        Recmodel,
        epoch,
        w=None,
        multicore=1,
        val=False) -> dict:
    pass

# ===============================================================================