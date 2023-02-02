# 관련 있는 것끼리 분류해봄
import torch
from torch import nn, optim
from torch import log
from sklearn.metrics import roc_auc_score

import numpy as np

from time import time

import world
from dataloader import BasicDataset
from model import GTN
from model import PairWiseModel

import random
import os

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

sample_ext = False

class BPRLoss:
    # 인자 recmodel은 PairWiseModel 인터페이스 타입으로 지정됨
    # bpr_loss 메소드가 반드시 정의돼 있어야 함을 의미함
    # 일전에 뜯어본 model.GTN 클래스는 bpr_loss를 정의해두었음
    def __init__(self,
                recmodel: PairWiseModel,
                config: dict) -> None:
        pass
    
    # GTN.bpr_loss 메소드를 이용해 mf_loss, reg_loss를 가져옴
    # loss는 mf_loss와 reg_loss의 합으로 계산됨
    # loss, mf_loss, reg_loss 모두 반환
    def stageOne(self, users, pos, neg) -> Tensor:
        pass

# ====================samplers================================

# sample_ext기 False이므로 _python 메소드가 실행됨
def UniformSample_original(dataset, neg_ratio=1) -> np.array:
    pass

# 주석으로 적은 걸 보면, LightGCN에서 BPR sampling을 구현한 것이라고 함
# 어쨌든 샘플링 작업?
def UniformSample_original_python(datset) -> np.array:
    pass

# =====================utils====================================

def set_seed(seed) -> None:     # seed값 설정
    pass

# world.args.gcn_model을 이용해서 파일을 가져옴
# 파이토치는 .pth 확장자를 사용하는 파일로 저장
# tar를 통해 압축하게 되면 *.pth.tar 확장자를 갖는 파일이 됨
# 아직 만들어진 걸 못 본 것 같음 → GTN에서는
# LightGCN lib에서는 checkpoints(FILE_PATH) 밑에 만들어졌음
def getFileName() -> str:
    pass

# 제너레이터가 뭔지는 잘 모르겠음
# 어쨌든 tensor가 1개만 인자로 들어오면 텐서 제너레이터를 반환
# 여러 개면 투플로 반환
def minibatch(*tensors, **kwarsg) -> Generator:
    pass

# 이름 그대로 섞은 결과값을 반환해주는 것 같음
def shuffle(*arrays, **kwargs) -> list:
    pass

# 정적 메소드는 클래스 이름과 닷 연산자로 바로 호출 가능
# 객체를 생성하고 생성된 객체로부터 호출할 필요가 없음
class timer:
    # 클래스 안에서 생성하는 건 처음 봄
    from time import time
    
    @staticmethod
    def get() -> int:
        pass

    def __init__(self) -> None:
        pass

    def __enter__(self) -> self:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

# ====================Metrics==============================

# bpr loss는 학습을 위해 사용되는 최적화 함수
# 평가 매트릭은 추천 성능을 확인하기 위한 알고리즘
    # 따라서 평가 매트릭은 학습을 좌우하지 않음
    # 학습을 좌우하는 것은 bpr loss

# RecallPrecision@K
# Metrics의 내부 원리까지는 지금 단게에서는 알 필요가 없음
def RecallPrecision_ATk(test_data, r, k) -> dict:
    pass

def MRRatK_r(r, k) -> float:
    pass

def NDCGatK_r(test_data, r, k) -> float:
    pass

def Auc(all_item_scores, dataset, test_data):
    pass

# 평가 매트릭 메소드이 인자 r을 생성하는데 사용됨
def getLabel(test_data, pred_data) -> np.array:
    pass