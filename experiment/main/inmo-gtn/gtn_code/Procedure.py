'''
## GTN-pytorch
"Graph Trend Filtering Networks for Recommendations", Accepted by SIGIR'2022.
Pytorch Implementation of GTN in Graph Trend Networks for Recommendations
The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch
'''




import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
CORES = multiprocessing.cpu_count() // 4

class AuxiliaryDataset():
    def __init__(self, dataset, user_map, item_map):
        self.n_users = len(user_map)
        self.n_items = len(item_map)
        self.device = world.device
        self.negative_sample_ratio = 1
        self.train_data = [[] for _ in range(self.n_users)]
        self.length = len(dataset)
        for o_user in range(dataset.n_users):
            if o_user in user_map:
                for o_item in dataset.train_data[o_user]:
                    if o_item in item_map:
                        self.train_data[user_map[o_user]].append(item_map[o_item])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        user = random.randint(0, self.n_users - 1)
        while not self.train_data[user]:
            user = random.randint(0, self.n_users - 1)
        pos_item = np.random.choice(self.train_data[user])
        data_with_negs = [[user, pos_item] for _ in range(self.negative_sample_ratio)]
        for idx in range(self.negative_sample_ratio):
            neg_item = random.randint(0, self.n_items - 1)
            while neg_item in self.train_data[user]:
                neg_item = random.randint(0, self.n_items - 1)
            data_with_negs[idx].append(neg_item)
        data_with_negs = np.array(data_with_negs, dtype=np.int64)
        return data_with_negs


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
        aux_sample = dataset.aux_dataloader(Recmodel.user_map, Recmodel.item_map)
    users = torch.Tensor(S[:, 0]).long()  # 41830
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1  # 21
    aver_loss = 0.
    aver_mf_loss = 0.0
    aver_reg_loss = 0.0

    '''
    tensor([[[20305, 32217, 40598]],

        [[ 5718, 23202, 30213]],

        [[14499, 22672, 31074]],

        ...,

        [[23634,  2750, 24274]],

        [[10288, 23247, 37299]],

        [[19462,   170, 28352]]])
    tensor([[[ 5997, 11856, 17133]],

        [[15768, 23682, 22133]],

        [[20036, 10689, 32474]],

        ...,

        [[16031, 22719, 18521]],

        [[24156, 19909,  2216]],

        [[ 6252, 14524,  7819]]])
    '''
    
    '''
    batch_data
    tensor([[[25448, 12367, 19837]],

            [[ 1699, 16913, 34226]],

            [[14856,  6121,  1717]],

            ...,

            [[ 8406,   821,  3345]],

            [[22039, 23720, 15303]],

            [[26734, 21682,  1290]]])

    a_batch_data
    tensor([[[25952, 20627,  5574]],

            [[16068, 27666, 38229]],

            [[16518, 19576, 13237]],

            ...,

            [[12104, 25097, 30376]],

            [[23307,  4702, 21889]],

            [[13053, 17841, 37854]]])
    '''

    for (batch_i,
         (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users,
                                                                           posItems,
                                                                           negItems,
                                                                           batch_size=world.config['bpr_batch_size'])):
        # cri, mf_loss, reg_loss = bpr.stageOne(batch_users, batch_pos, batch_neg)
        # users: tensor([ 9187, 22958, 24374,  ..., 11197, 23202, 21103], device='cuda:0')
        cri, mf_loss, reg_loss, learning_model = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        aver_mf_loss += mf_loss
        aver_reg_loss += reg_loss
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    aver_mf_loss = aver_mf_loss / total_batch
    aver_reg_loss = aver_reg_loss / total_batch

    # ===============================
    learning_model.feat_mat_anneal()  
    # ===============================   

    time_info = timer.dict()
    timer.zero()
    return f"loss {aver_loss:.4f}  {aver_mf_loss:.4f}  {aver_reg_loss:.4f} - {time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, w=None, multicore=1, val=False):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    # valDict: dict = dataset.valDict

    Recmodel: model.GTN
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)  # can speed up: self._allPos
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)   
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            # posivite instances
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()

        return results