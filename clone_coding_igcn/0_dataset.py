# 사용자 정의 모듈 X
# 진출 차수 0
import os
import numpy as np
from torch.utils.data import Dataset
import random
import sys
import time
import json

def get_dataset(config) -> dataset:
    pass

# 유저-아이템 집합을 업데이트?
# user_inter_sets와 item_inter_sets를 업데이트
def update_ui_sets(u, i, user_inter_sets: dict, item_inter_sets: dict) -> None:
    pass

# 유저의 관계 리스트를 업데이트?
# user_inter_lists를 업데이트
def update_user_inter_lists(u, i, t, user_map, item_map, user_inter_list) -> None:
    pass

# 파일을 열어서 write
def output_data(file_path, data) -> None:
    pass


class BasicDataset(Dataset):
    # torch.utils.data 라이브러리의 Dataset 클래스를 상속받음
    def __init__(self) -> None:
        super().__init__()

    # 유저 관계 집합과 아이템 관계 집합을 가지고 맵을 반환?
    # 스파스 유저-아이템을 제거한다는 의미는?
    def remove_sparse_ui(self, user_inter_sets, item_inter_sets) -> user_map, item_map:
        pass

    # train, val, test 데이터 생성 → 틀만 만드는 것 같음
    # 자식 클래스에서 호출됨
    def generate_data(self) -> None:
        pass

    def __len__(self) -> train_array_length:
        pass

    # negative 아이템을 왜 얻지?
    def __getitem__(self, index) -> data_with_negs:
        pass

    # output_data 메소들 이용해서 파일에 write
    def ouput_dataset(self, path) -> None:
        pass


class ProcessedDataset(BasicDataset):
    def __init__(self) -> None:
        super().__init__()
    
    # 데이터를 읽어서 실질적인 train, test, val 데이터를 만드는 것 같음
    def read_data(self, file_path) -> data:
        pass


# 아래 메소드들이 모두 호출됨
# update_ui_sets(), update_user_inter_lists(), remove_sparse_ui(),  generate_data()
# update 메소드는 iterable 타입을 인자로 받으므로 call by referecne 방식으로 호출됨
class GowallaDataset(BasicDataset):
    def __init__(self) -> None:
        super().__init__()


# 이건 뭔데 trainer 모듈에서 쓰이지?
class AuxilliaryDataset(BasicDataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> train_array_length:
        return super().__len__()
    