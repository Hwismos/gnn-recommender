import random
import world
from dataloader import BasicDataset     # 왜 Loader 클래스를 임포트하지 않았을까?
import dataloader
from gtn_propagation import GeneralPropagation

# torch.nn: 그래프를 위한 기본 빌딩 블록
import numpy as np
import warnings
import torch
from torch import nn


from torch_sparse import SparseTensor
from torch_sparse import SparseTensor
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

seed = 2020

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# torch.nn 라이브러리의 Module 클래스를 상속 받음
# 객체 생성이 끝
# 객체 생성하며 부모 클래스 객체도 같이 생성


class BasicModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # 얘도 반드시 오버라이딩 해야 함
    def getUserRating(self, users):
        raise NotImplementedError

# ! 생각해보니 인터페이스


class PairWiseModel(BasicModel):
    def __init__(self) -> None:
        super().__init__()

    # 얘도 반드시 오버라이딩 해야 함
    # 인자로 유저 리스트와 각 유저에 대한 pos, neg itmes를 받아옴
    # log 로스와 l2 로스를 반환
    # l2 loss가 reg loss인 것 같음
    # float를 반환하지 않을까 추측함
    # 되도록이면 하나의 타입(단일 변수)만 반환하는 것이 좋은 것 같음
    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError


# * 전파(propagation)를 외부 모듈에서 하니 여기서는 임베딩 생성만?
class GTN(BasicModel):
    # 언더바 두 개는 초기화될 때, 언더바 하나는 모듈 내에서만 사용되는 것 같음
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset,
                 args) -> None:
        super().__init__()
        self.__init_weight()
        # args가 어떻게 들어오는지 모르겠음
        # 객체형태로 들어오는 것 같기도 함
        self.gp = GeneralPropagation(
            args.K, args.alpha, cached=True, args=args)

    # ! 객체 생성할 때 만들어지는 걸 보니 임베딩 테이블 만드는 것 같음
    # pretrain 돼 있으면 기존 임베딩을 이용하는 것 같음
    # 아니면 정규분포 쓰는 듯
    def __init_weight() -> None:
        pass

    # keep_prob는 배치 사이즈와 관련된 arg임
    # _dropout 메소드로부터 참조됨
    def __dropout_x(self, x, keep_prob) -> Tensor:
        pass

    # gowalla처럼 큰 데이터에 대한 인접행렬을 분해하기 위해 쓰임
    # 어쨌든 그래프를 반환함
    # 그래프는 리스트 형태
    def __dropout(self, keep_prob) -> Tensor:
        pass

    # 임베딩 가중치를 초기화(임베딩 테이블 생성)한 뒤 전파
    # ! gtn_propagation.py 모듈의 forward 메소드가 사용됨
    # 유저와 아이템의 초기 임베딩을 합친 텐서와 간선 가중치(텐서)를 이용해서 forward 메소드 콜
    # ! edge_index(간선 가중치) → 인접행렬
    # * 학습된 유저, 아이템의 임베딩을 반환
    def computer(self, corrupted_graph=None) -> Tensor:
        pass

    # 학습된 유저, 아이템 임베딩을 내적해서 rating을 계산
    def getUserRating(self, users) -> Tensor:
        return super().getUserRating(users)

    # 역시 학습된 임베딩 사용
    # bpr loss를 계산하기 위해 호출됨
    # ego는 초기 임베딩인 것 같음
    def getEmbedding(self, users, pos_items, neg_items):
        pass

    # 논문의 수식과 어떻게 일치하는지는 잘 모르겠지만
    # log_loss가 ln 부분인 것 같고 reg_loss가 E_{in}부분인 것 같음
    def bpr_loss(self, users, pos, neg) -> float:
        pass

    # gamma가 어디에 쓰이는지 모르겠음
    def forward(self, users, items) -> Tensor:
        pass
