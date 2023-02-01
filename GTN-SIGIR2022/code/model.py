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


import igcn_copy
import random
import world
from dataloader import BasicDataset
import dataloader
from gtn_propagation import GeneralPropagation

# torch.nn: 그래프를 위한 기본 빌딩 블록
import numpy as np
import warnings
import torch
from torch import nn


from torch_sparse import SparseTensor
from torch_sparse import SparseTensor
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

# In[2]:
seed = 2020

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)


# nn.Module 클래스를 인자로 전달받기 때문에 train 등의 메소드(파일 내에서는 정의되지 않은)를 사용할 수 있는 것으로 추측
# BasicModel 클래스를 PairWiseModel, GTN 클래스가 상속 받음
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
        # gtn 객체를 초기화할 때 dataset 필드의 타입 힌트를 BasicDataset 클래스로 설정
        self.dataset: dataloader.BasicDataset = dataset
        self.args = args
        self.__init_weight()

        # gtn 객체를 초기화할 때 gp 필드를 gtn_propagation 모듈의 GeneralPropagation 객체로 초기화
        self.gp = GeneralPropagation(
            args.K, args.alpha, cached=True, args=args)

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.args.K
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        
        
        # # * 여기 임베딩을 바꿔줘야 학습이 진행될 수 있을 것 같음
        # # ! ===================================================================================================
        # with open('/home/hwiric/Internship/GTN-SIGIR2022/code/igcn_copy/log.txt', 'r') as f:
        #     flag=False
        #     all_emb=[]
        #     for line in f:
        #         line=''.join(line.rstrip('\n'))
        #         if flag==False:
        #             if line!='START':
        #                 continue
        #             else:
        #                 flag=True
        #         else:
        #             # line=line[1:-1].split(', ')
        #             if line=='END':
        #                 continue
        #             line=list(map(float, line[1:-1].split(', ')))
        #             all_emb.append(line)
        # all_emb=torch.FloatTensor(all_emb).to(world.device)
        # # ! ===================================================================================================

        # ? ======================================ORIGINAL======================================================
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, 
            embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, 
            embedding_dim=self.latent_dim)

        # ? ===================================================================================================
        
        # ? ===========================================연결======================================================
        # import igcn_copy

        # final_rep=igcn_copy.main()

        # all_emb=torch.split(final_rep, [self.num_users, self.num_items])
        # e_user, e_item=all_emb[0], all_emb[1]
        
        # emb_user=nn.Embedding.from_pretrained(e_user, freeze=False)
        # emb_item=nn.Embedding.from_pretrained(e_item, freeze=False)
        

        # self.embedding_user = emb_user
        # self.embedding_item = emb_item
        # ? ===================================================================================================
        
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb']))
            # pretrained?
            print('use pretarined data')
        # 객체의 f(function) 변수에 sigmoid 함수 저장
        self.f = nn.Sigmoid()
        # SparseGraph를 만들어서 객체의 Graph 변수에 저장하고자 함
        self.Graph = self.dataset.getSparseGraph()

        print(f"lgn is already to go(dropout:{self.config['dropout']})")

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

    def computer(self, corrupted_graph=None):
        """
        propagate methods for lightGCN
        """

        # print('\n##########################################################################################\n')
        # # # print(type(self.embedding_user))    # <class 'torch.nn.modules.sparse.Embedding'>
        # # # # print(self.embedding_user.shape)
        # # # print('\n\n\n')
        # # # print(type(self.embedding_user.weight))     # <class 'torch.nn.parameter.Parameter'>
        # # print(self.embedding_user.weight.shape)   # torch.Size([458, 64])
        # # print(self.embedding_item.weight.shape)     # torch.Size([1605, 64])            # 총합 2063
        # print('\n##########################################################################################\n')
        # exit()


        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])     # ! 여기에 붙이면 됨 → 아님

        # <class 'torch.Tensor'>
        # torch.Size([2063, 64])

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
        edge_index = SparseTensor(row=r,
                                    col=c,
                                    value=g_droped.values(),
                                    sparse_sizes=(num_nodes, num_nodes))
        
        emb, embs = self.gp.forward(x, edge_index, mode=self.args.gcn_model)
        light_out = emb

        '''
        # ! seed값이 동일해서 임베딩 룩업 테이블은 LightGCN(lgcn_lib)과 정확히 같음
        tensor([[ 27.2716,  -0.9661,  15.4678,  ..., -11.3022,  24.2125,   0.0000],
        [ -0.2903,  -0.1734,  -0.0956,  ...,   0.0739,   0.0000,  -0.1329],
        [ 38.0587,  28.5872, -34.5981,  ..., -29.6428, -27.6292,  -8.7488],
        ...,
        [  1.1554,  -0.3685,  -0.7906,  ...,   0.0000,   0.7672,  -0.4871],
        [  0.4570,  -1.2098,   0.3067,  ...,  -0.5867,   1.4079,  -1.2274],
        [  1.1645,  -0.2060,   0.0000,  ...,   0.0000,  -0.9328,  -0.7533]],
        '''
        # print(f'===========================\n{light_out}\n===========================')
        # exit()

        # 합친 뒤 학습시키고 분리하는 것 같음
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        # print(f'유저 임베딩: {users_emb}')
        # print(f'유저 임베딩 shape: {users_emb.shape}')
        # print(f'아이템 임베딩: {items_emb}')
        # print(f'아이템 임베딩 shape: {items_emb.shape}')
        # exit()

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        # print(f'===========================\n{all_users}\n===========================')
        # exit()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
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

        loss = torch.mean(torch.nn.functional.softplus(
            neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
