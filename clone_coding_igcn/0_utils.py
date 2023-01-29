# 여기도 사용자 모듈 없음
import numpy as np
import torch
import random
import os
import sys
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import networkx as nx
from sortedcontainers import SortedList

def set_seed(seed) -> None:
    pass

# 언제 닫지?
def init_run(log_path, seed) -> None:
    pass

# mat이 어떤 타입으로 오는지 알아야 함
# mat을 tocoo() 메소드르 이용해 coo format으로 변경
def get_sparse_tensor(mat, device) -> sparse_tensor:
    pass

# 데이터셋을 이용해 인접 매트릭스를 반환
# 왜 utils에 있지?
# dataset에 없고...?
def generate_daj_mat(dataset) -> adj_mat:
    pass

# 랭킹 메트릭에 따라 연산된 ranked 유저와 아이템을 반환
# degree, greedy, page_rank
# 논문에서는 뭐 썼지?
# 아무것도 없으면 그냥 정렬?
def graph_rank_nodes(dataset, ranking_metric): # -> ranked_users, ranked_items
    pass


# 이건 뭐지...?
class AverageMeter:
    def __init__(self) -> None:
        pass

    def update(self, val, n=1) -> None:
        pass


# object 클래스를 상속 받음
# 객체를 생성할 때 stream을 받는데 뭔지는 모르겠음
class Unbuffered(object):
    def __init__(self, stream) -> None:
        pass

    def write(self, data) -> None:
        pass 

    def write_lines(self, datas) -> None:
        pass

    def __getattr__(self, attr: str) -> Any:
        pass