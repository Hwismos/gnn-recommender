from typing import Optional, Tuple  # 뭐하는 패키지인지 모르겠음
# typing 모듈과 연관이 있는 것 같음
# Adj는 인접행렬(그래프)인 것 같음
from torch_geomtetric.typing import Adj, OptTensor  

# 이쯤에서 torch가 무엇인지에 대한 정리
# 파이썬을 염두에 두고 머신러닝/딥러닝을 위해 설계된 오픈 소스 라이브러리
# 텐서 계산, gpu 가속 등의 역할을 함
import torch
from torch import Tensor            # Tensor 패키지 임포트
import torch.nn.functional as F     # 시크모이드와 같은 비선현 함수를 이용하기 위해 임포트한 것 같음
from torch_sparse import SparseTensor, matmul       # sparse한 텐서를 만들기 위한 것 같음
# torch.nn은 그래프를 위한 기본 빌딩 블록
# torch_geometric은 GNN을 위해 설계된 라이브러리임
# 고로 아래 있는 모듈들은 모두 GNN 전용 라이브러리라는 것
from torch_geometric.nn.conv import MessagePassing  
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn       # 얘도 그래프를 위한 라이브러리

# 똑같은 거 두 번 나왔음
# 실수인가?
# import torch.nn.functional as F

import numpy as np
import time

seed = 2020
import random
import numpy as np

# 여기서도 난수 생성
# 왜 항상 난수가 필요한거지?
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# 하나의 클래스만 있음
# MessagePassing 객체를 상속 받음
# 따라서 MessagePassing 객체의 필드, 메소드들을 사용할 수 있음
# GP 클래스에 정의되지 않은 필드, 메소드들은 모두 상속받은 필드, 메소드들임 
class GeneralPropagation(MessagePassing):
    # 언더바로 시작되는 변수는 클래스인가 모듈 내에서만 쓰는 걸로 알고 있음
    # _cached 변수들은 gtn_propagation.py 모듈 내에서만 쓰이는 것 같음
    # 다 torch 라이브러리를 사용함
    # 뭔지는 잘 모르겠음
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cahced_adj_t: Optional[SparseTensor]
    
    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 add_self_loops_l1: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 num_classes: int = None,
                 args = None,
                 **kwargs):
        # 객체를 만들 때 디폴트로 지정되는 값들이 많음
        # **은 딕셔너리 타입을 인자로 받을 때 사용

        # 부모 클래스 객체 생성
        # MessagePassing 객체를 생성할 때 aggregation 함수를 add로 설정하는 것 같음
        super(GeneralPropagation, self).__init__(aggr='add', **kwargs)

        # 객체의 필드 초기화
        self.K = K
        self.alpha = alpha
        self.mode = mode
        self.dropout=args.prop_dropout      # args의 default type이 None이기 때문에 클래스 타입이 와도 상관 없음
        self.cached = cached
        self.add_self_loops=add_self_loops
        self.add_self_loops_l1=add_self_loops_l1
        
        self.normalize=normalize
        
        # 이럴거면 뭐하러?
        self._cached_edge_index=None
        self._cahced_adj_t=None

        self.node_num=node_num
        self.num_classes=num_classes
        self.args=args
        self.max_value=None
        self.debug=self.args.debug      # args에 뭐가 들어오는거지?

    # 매개변수를 초기화함
    # 다 None 타입으로 초기화
    # 초기화하는 변수들이 어떤 역할을 하는지는 잘 모르겠음
    # 간선의 인덱스, 인접행렬, 입사행렬(incident matrix)를 초기화
    def reset_parameters(self) -> None: 
        pass

    # 입사행렬을 반환함
    def get_incident_matrix(self, edge_index: Adj) -> Tensor:
        pass

    # 입사행렬을 정규화함
    # 정규화된 입사행렬 반환
    def inc_norm(self, inc, edge_index, add_self_loops, normalize_para=-0.5) -> Tensor:
        pass

    # 입사행렬을 검사하는 것 같음
    # 문제가 있음은 assert문 작동
    def check_inc(self, edge_index, inc, normalize=False) -> None:
        pass

    # -> Tensor 가 Tensor를 반환하는 힌트인걸로 알고 있음
    # 딱 봐도 어려워보임
    # 중요한 건 이것저것 받아서 텐서 타입을 반환한다는 점
    # gtn_forward 메소드를 이용하고, gtn_forward 메소드는 proximal... 메소드를 이용함
    # 3개가 쭈루룩 연결됨
    # 두 값을 반환하는데 niter(iteration number)에 연관 있는 것 같음
    def forward(self, 
                x: Tensor, 
                edge_index: Adj, 
                x_idx: Tensor = None, 
                edge_weight: OptTensor = None, 
                mode=None, 
                niter=None,
                data=None) -> Tensor:
        pass

    # forward 메소드에서 참조됐음
    # 논문에서도 패스했으니 여기서도 패스
    # 수학적 사고를 구현했다는 것만 앎
    def gtn_forward(self, x, hh, K, incident_matrix) -> Tensor:
        pass

    # 3-step으로 만드는 수학 공식 부분을 구현한 것 같음
    # gtn_forward 메소드에서 참조됨
    # 얘도 패스
    def proximal_l1_conjugate(self, x: Tensor, lambda2, beta, gamma, m) -> Tensor:
        pass

    # 이쯤에서 tensor의 개념을 복습
    # 텐서는 매트릭스의 집합
    # 매트릭스는 벡터의 집합
    # 벡터는 스칼라의 집합!!!
    
    # 다른 노드들로부터 오는 메시지를 구성하는 역할을 하는 것 같음
    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        pass

    # 메시지들을 종합하는 역할을 함
    # 역시 텐서 타입을 반환
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        pass

    # representation
    # 사용자가 이해할 수 있는 객체의 모습을 표현 → 설명서
    # 스트링의 format 메소드를 사용함
    def __repr__(self):
        return '{}(K={}, alpha={}, mode={}, dropout={})'.format(self.__class__.__name__,
                                                                self.K,
                                                                self.alpha,
                                                                self.mode,
                                                                self.dropout)                                                              )




