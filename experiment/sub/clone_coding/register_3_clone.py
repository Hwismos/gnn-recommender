import world
import dataloader
import model
import utils

from pprint import pprint

import random
import numpy as np
import torch

seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# 메소드, 클래스가 없는 단순 모듈

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)    # default가 gtn
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")             # bpr loss를 이용해 학습
print('===========end===================')

# value가 클래스 타입
# MODELS 딕셔너리를 이용해 바로 객체를 생성할 수 있음
MODELS = {
    'gtn': model.GTN,
    'lgn': model.LightGCN,
}
