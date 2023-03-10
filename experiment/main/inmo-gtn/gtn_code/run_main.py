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

seed = 2021
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

Recmodel = register.MODELS[world.model_name](world.config, dataset, world.args)
Recmodel = Recmodel.to(world.device)

bpr = utils.BPRLoss(Recmodel, world.config)

baisc_path_log = "../data/" + str(world.args.dataset) + "/log/"

weight_file = utils.getFileName()
# print(f"load and save to {weight_file}")

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard

w = None
world.cprint("not enable tensorflowboard")

final_topk_txt = ""
try:
    timeStamp = time.time()

    print(world.args)
    import matplotlib.pyplot as plt

    y = []
    x = list(range(world.TRAIN_epochs))
    plt.figure()

    early_stop_cnt = 0
    best_ndcg = -987654321

    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)

        if epoch != 0 and epoch % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)

            pre = round(results['precision'][0], 5)
            recall = round(results['recall'][0], 5)
            ndcg = round(results['ndcg'][0], 5)

            topk_txt = f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}] | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
            print(topk_txt)

            if ndcg == best_ndcg:
                early_stop_cnt += 1

                if early_stop_cnt == 10:
                    print('Early Stopping')
                    break

            if best_ndcg < ndcg:
                best_ndcg = ndcg
                print(f'Best NDCG is {best_ndcg}')

        end = time.time()
        diff = datetime.datetime.fromtimestamp((end - start))
        diff_mins = diff.strftime("%M:%S")
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}  |  {diff_mins}mins')


finally:
    if world.tensorboard:
        w.close()