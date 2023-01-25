'''
## GTN-pytorch
"Graph Trend Filtering Networks for Recommendations", Accepted by SIGIR'2022.
Pytorch Implementation of GTN in Graph Trend Filtering Networks for Recommendations
The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch

@inproceedings{fan2022graph,
  title={Graph Trend Filtering Networks for Recommendations},
  author={Fan, Wenqi and Liu, Xiaorui and Jin, Wei and Zhao, Xiangyu and Tang, Jiliang and Li, Qing},
  booktitle={International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2022}
}
'''

import world
import dataloader
import model
import utils
from pprint import pprint

seed = 2020
import random
import numpy as np
import torch

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'last-fm']:
    # dataset = dataloader.Loader(path="../data/" + world.dataset)

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    # lastfm 로딩 객체 추가
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

# Model 사전 자료형
# key는 gtn, value는 model 모듈의 GTN 클래스
MODELS = {
    'gtn': model.GTN,
    'lgn': model.LightGCN,
}
