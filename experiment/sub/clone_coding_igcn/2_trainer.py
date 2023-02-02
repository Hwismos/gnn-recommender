import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
import torch.nn.functional as F
import scipy.sparse as sp

import sys
import time
import numpy as np

import os
from 0_utils import AverageMeter, get_sparse_tensor
from 0_datasets import AuxiliaryDataset

class BasicTrainer:
    def __init__(self, trainer_config) -> None:
        pass

    # 최적화기 초기화
    def initialize_optimizer(self) -> None:
        pass

    # 인터페이스
    def train_one_epoch(self):  
        # -> loss.avg
        raise NotImplementedError

    # tensorboard 같은 곳에 기록하는 것 같음
    def record(self, writer, stage, metrics) -> None:
        pass

    # train_one_epoch 메소드를 호출함
    # loss를 줄이는 방향으로 학습하고 최종적으로는 평가 메트릭(ndcg)으로 성능을 평가
    # verbose는 함수 수행시 자세한 정보를 출력할지 말지를 결정함
    def train(self, verbose=True, writer=None):     
        # -> ndcg
        pass

    # 평가 메트릭 결과를 반환
    def calculate_metrics(self, eval_data, rec_items):
        # -> results{'Precision': x, 'Recall': y, 'NDCG': z}
        pass
    
    # cal_metric과 뭐가 다른거지?
    def eval(self, val_or_test, banned_items=None):
        # -> results, metrics
        pass
    
    # 이건 뭐지?
    def inductive_eval(self, n_old_users, n_old_items) -> None:
        pass


# IGCN이 학습할 때 사용하는 최적화 함수
class IGCNTrainer(BasicTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
    # utils 모듈의 AverageMeter 클래스 타입의 객체 생성
    # BPRTrainer에서도 쓰이네
    # 여기서도 loss.backward() 호출됨 → 당연하지, 역전파되야 학습이 되니까
    def train_one_epoch(self):      # -> loss.avg
        # loss = bpr_loss + reg_loss
        return super().train_one_epoch()