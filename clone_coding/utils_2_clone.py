import torch
from torch import nn, optim
import numpy as np
from torch import log
from time import time
import world
from dataloader import BasicDataset
from model import GTN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)