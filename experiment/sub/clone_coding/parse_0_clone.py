import argparse # 입력 받기 위한 모듈
import torch    # 메인 페키지, pytorch

# seed 값을 이용해 난수를 생성하는 걸로 알고 있음
seed = 2020
import random
import numpy as np

# 임베딩 룩업 테이블을 위해 난수를 생성하는 부분이 아닐까 싶음
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# 함수가 하나 밖에 없음
# 얘만 호출됨
# 이 함수는 인자를 파싱해서 반환하는 역할을 함
# 적어보니 확실히 알겠음
def parse_args():
    parser = argparse.ArgumentParser(description="Go GTN")  # 시작 부분인 것 같음
    
    # 나머지는 하이퍼파라미터를 조정하는 옵션 부
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")  # 512 1024 2048 4096
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    
    parser.add_argument('--epochs', type=int, default=100)  # 1000, ...

    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing, 100")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="gtn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    parser.add_argument('--prop_dropout', type=float, default=0.1)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--ogb', type=bool, default=True)
    parser.add_argument('--incnorm_para', type=bool, default=True)

    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1)

    parser.add_argument('--alpha1', type=float, default=0.25)
    parser.add_argument('--alpha2', type=float, default=0.25)

    parser.add_argument('--lambda2', type=float, default=4.0) #2, 3, 4,...

    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate:0.001")  # 0.001
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [gowalla,  lastfm, yelp2018, amazon-book]")
    parser.add_argument('--model', type=str, default='gtn', help='rec-model, support [gtn, lgn]')
    parser.add_argument('--avg', type=int, default=0)
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of GTN: 128, 256")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gcn_model', type=str,
                        default='GTN', help='GTN')

    return parser.parse_args()  # arguments를 parsing해서 반환