import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
# from utils import get_sparse_tensor, graph_rank_nodes, generate_daj_mat
from torch.nn.init import kaiming_uniform_, xavier_normal, normal_, zeros_, ones_
import sys
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.utils.checkpoint import checkpoint
import dgl
import multiprocessing as mp
import time

# utils 모듈에서 메소드들 가져와서 사용함
from utils import get_sparse_tensor, graph_rank_nodes, generate_daj_mat

# 사용할 GNN 모델 반환
def get_model(config, dataset): # -> model
    pass

# 한 레이어를 초기화한다고 이름은 쓰여 있음
# 인자와 반환 값 타입을 확인해보면 더 확실해질 듯함
def init_one_layer(in_features, out_feature): # -> layer
    pass

class BasicModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    # 이거는 반드시 구현해야 함
    # 에러 걸려 있음
    def predict(self, users):
        pass

    # 아래 두 메소드는 정의돼 있음
    # 걍 쓰면 될 듯
    def save(self, path):
        pass

    def load(self, path):
        pass


class IGCN(BasicModel):
    def __init__(self) -> None:
        super().__init__()

    def update_feat_mat(self) -> None:
        pass

    # annealing
    def feat_mat_anneal(self) -> None:
        pass

    def generate_graph(self, dataset): # -> norm_adj
        pass

    # utils.graph_rank_nodes() 호출
    # 랭킹 메트릭 None으로 설정함
    # 반환 값들을 어떻게 쓰는거지?
    # 애초에 map이 뭘까 → 임베딩?
    def generate_feat(self, 
                        dataset, 
                        is_updating=False, 
                        ranking_metric=None):   
                        # -> feat, user_map, item_map, row_sum
        pass

    def inductve_rep_layer(self, feat_mat): # -> representations
        pass

    # 여기서는 학습 시키고
    def get_rep(self):  # -> final_representations
        pass

    # 여기서는 bpr을 전파시키는 건가?
    # 그래서 학습 시킬 때 이걸 뒤로 보내고?
    def bpr_forward(self, users, pos_items, neg_items): 
        # -> users_r, pos_items_r, neg_items_r, l2_norm_sq
        pass

    def predict(self, users):   # -> scores
        return super().predict(users)

    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass