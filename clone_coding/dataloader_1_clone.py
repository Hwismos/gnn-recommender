import numpy as np
import random
import pandas as pd     # 데이터 조작 및 분석을 위한 라이브러리
from torch.utils.data import Dataset, DataLoader

from scipy.sparse import csr_matrix     # 스파스 매트릭스를 저장하는 방법
import scipy.sparse as sp

import world
from world import cprint

from time import time

from sklearn.model.selection import train_test_split

seed = 2020

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# Dataset 클래스를 상속 받음


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    # 부모 클래스(이 클래스)를 상속 받을 때 반드시 오버라이딩해야 하는 필드와 메소드
    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

# 파이토치 라이브러리를 위한 데이터셋 구상
# 그래프 정보도 포함돼 있다고 함


class Loader(BasicDataset):
    def __init__(self,
                 config=world.config,
                 path="../data/gowalla".
                 flag_test=0):
        pass

    def random_sample_edges(self, adj, n, exclude):
        pass

    # 보통 참조하는 메소드(caller)를 위에 쓰나봄
    # random_sample_edges에서 참조됨
    def sample_forever(self, adj, exclude):
        pass

    # 부모 클래스에서 오버라이딩을 강제했던 필드들을 재정의
    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def valDict(self):
        return self.__valDict

    @property
    def allPos(self):
        return self._allPos

    # A_fold가 어디에 쓰이는지 모르겠음 → 어디에서 안 쓰임...?
    # _convert_sp_mat_to_sp_tensor 메소드 콜
    # 텐서를 원소로 하는 리스트?
    def _split_A_hat(self, A) -> list:
        pass

    # 스파스 매트릭스를 스파스 텐서로 변환
    def _convert_sp_mat_to_sp_tensor(self, X) -> Tensor:
        pass

    # 오버라이딩 해야하는 메소드
    # 스파스 그래프 생성 → 그래프 정보를 포함한다는게 이거인 것 같음
    # 객체의 인접행렬 초기화 함
    # 텐서 타입으로 반환하는 것 같음
    # * 그래프를 인접행렬로 생각한다면, 그리포 텐서를 매트릭스의 집합으로 생각한다면
    # * 하나의 매트릭스 역시 텐서가 되므로 그래프를 인접행렬로 표현했다는 추측이 맞을 수 있음
    def getSparseGraph(self) -> Tensor:
        return super().getSparseGraph()

    # {user: [item]} 형태의 테스트 데이터셋 반환
    def __build_test(self) -> dict:
        pass

    # {user: [items]} 형태의 valid 데이터셋 반환
    def __build_val(self) -> dict:
        pass

    # np.array 타입을 reshape 해서 반환
    # 어떤 역할인지는 잘 모르겠음
    def getUserItemFeedback(self, users, items) -> np.array:
        return super().getUserItemFeedback(users, items)

    # getuserNegItems는 안해도 써 있었음
    # 유저와 관계있는, positive한 아이템 리스트를 반환
    def getUserPosItems(self, users) -> list:
        return super().getUserPosItems(users)
