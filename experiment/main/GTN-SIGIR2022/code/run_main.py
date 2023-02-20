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

# ! local 아래에 있는 packages 에서 torch를 찾지 않도록 세팅
import sys
sys.path.remove('/home1/prof/hwang1/.local/lib/python3.8/site-packages')

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time, datetime
import Procedure
from os.path import join

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# ==================kill============
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

if world.model_name == 'gtn':
    Recmodel = register.MODELS[world.model_name](world.config, dataset, world.args)
else:
    Recmodel = register.MODELS[world.model_name](world.config, dataset)

# cprint(world.model_name)

Recmodel = Recmodel.to(world.device)

bpr = utils.BPRLoss(Recmodel, world.config)

baisc_path_log = "../data/" + str(world.args.dataset) + "/log/"

weight_file = utils.getFileName()

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
# w = None
# world.cprint("not enable tensorflowboard")

# 아래 코드가 없이 w에 None 타입의 객체만 저장해두고 있었음
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

final_topk_txt = ""
try:
    timeStamp = time.time()

    # import matplotlib.pyplot as plt

    y = []
    x = list(range(world.TRAIN_epochs))
    # plt.figure()
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)

            pre = round(results['precision'][0], 5)
            recall = round(results['recall'][0], 5)
            ndcg = round(results['ndcg'][0], 5)

            topk_txt = f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}] | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
            print(f'\n{topk_txt}\n')

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        
        end = time.time()
        diff = datetime.datetime.fromtimestamp((end - start))
        diff_mins = diff.strftime("%M:%S")
        
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}  |  {diff_mins}mins')

finally:
    if world.tensorboard:
        w.close()
