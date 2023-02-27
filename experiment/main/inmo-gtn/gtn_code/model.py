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
from dataloader import BasicDataset
from gtn_propagation import GeneralPropagation

import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
import dgl
from torch.nn.init import normal_

import warnings
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric
from torch_sparse import SparseTensor


# In[2]:
seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class GTN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset, args):
        super(GTN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.args = args
        self.__init_weight()

        self.gp = GeneralPropagation(args.K, args.alpha, cached=True, args=args)

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.args.K
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        '''
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')
        '''
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # ================================INMO_FIELD==================================
        self.dropout = 0.
        self.trainable = True
        self.device = world.device
        self.feature_ratio = 1.
        self.alpha = 1.
        self.delta = 0.99
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(self.dataset,
                               ranking_metric='sort')
        self.update_feat_mat()

        self.embedding = torch.nn.Embedding(self.feat_mat.shape[1], self.latent_dim)
        # self.embedding = torch.nn.Embedding(self.feat_mat.shape[0], self.latent_dim)
        self.w = torch.nn.Parameter(torch.ones([self.latent_dim], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        # ============================================================================

    # ==============================INMO_METHOD====================================
    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()
    
    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        if not is_updating:
            # feature_ratio를 1로 가정
            core_users = np.arange(self.num_users, dtype=np.int64)
            core_items = np.arange(self.num_items, dtype=np.int64)

            # if self.feature_ratio < 1.:
            #     ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
            #     core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
            #     core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            # else:
            #     core_users = np.arange(self.n_users, dtype=np.int64)
            #     core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.num_users + item, user_map[user]])
        for user in range(self.num_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.num_items):
            indices.append([self.num_users + item, user_dim + item_dim + 1])

        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                            shape=(self.num_users + self.num_items, user_dim + item_dim + 2), 
                            dtype=np.float32).tocsr()


        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = self.get_sparse_tensor(feat, self.device)
                    
        return feat, user_map, item_map, row_sum
    
    # inmo 패키지의 utils에 있던 메소드
    def get_sparse_tensor(self, mat, device):
        coo = mat.tocoo()
        indexes = np.stack([coo.row, coo.col], axis=0)
        indexes = torch.tensor(indexes, dtype=torch.int64, device=device)
        data = torch.tensor(coo.data, dtype=torch.float32, device=device)
        sp_tensor = torch.sparse.FloatTensor(indexes, data, torch.Size(coo.shape)).coalesce()
        return sp_tensor

    def inductive_rep_layer(self, feat_mat):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.latent_dim],
                                    dtype=torch.float32, device=self.device)
        # padding_tensor = torch.empty([0, self.latent_dim],
        #                             dtype=torch.float32, device=self.device)


        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()

        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]

        return x
    # ============================================================================

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    # inmo method
    def dropout_sp_mat(self, mat):
        if not self.training:
            return mat
        random_tensor = 1 - self.dropout
        random_tensor += torch.rand(mat._nnz()).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = mat.indices()
        v = mat.values()

        i = i[:, dropout_mask]
        v = v[dropout_mask] / (1. - self.dropout)
        out = torch.sparse.FloatTensor(i, v, mat.shape).coalesce()
        return out

    def computer(self, corrupted_graph=None):
        """
        propagate methods for lightGCN
        """
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        # all_emb = torch.cat([users_emb, items_emb])

        feat_mat = self.dropout_sp_mat(self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)
        all_emb = representations
        
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                if corrupted_graph == None:
                    g_droped = self.Graph
                else:
                    g_droped = corrupted_graph
        else:
            if corrupted_graph == None:
                g_droped = self.Graph
            else:
                g_droped = corrupted_graph

        # our GCNs
        x = all_emb
        rc = g_droped.indices()
        r = rc[0]
        c = rc[1]
        num_nodes = g_droped.shape[0]
        edge_index = SparseTensor(row=r, col=c, value=g_droped.values(), sparse_sizes=(num_nodes, num_nodes))
        emb, embs = self.gp.forward(x, edge_index, mode=self.args.gcn_model)
        light_out = emb

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        # users_emb_ego = self.embedding_user(users)
        # pos_emb_ego = self.embedding_item(pos_items)
        # neg_emb_ego = self.embedding_item(neg_items)
        users_emb_ego = self.embedding(users)
        pos_emb_ego = self.embedding(self.num_users + pos_items)
        neg_emb_ego = self.embedding(self.num_users + neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma