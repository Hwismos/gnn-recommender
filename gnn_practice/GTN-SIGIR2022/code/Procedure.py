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

# observed interaction에 대한 prediction score가 unobserved interaction보다 높은 값을 가지는 것을 가정해 파라미터들을 최적화
def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    # 인자로 받아온 값을 이용해 Recmodel(추천 모델)을 초기화
    Recmodel = recommend_model
    # torch.nn 모듈 덕분에 train 메소드가 동작 가능한 것으로 추측
    # 모델 훈련
    Recmodel.train()
    # bpr 손실함수를 이용하는 학습에서 bpr 필드를 utils 모듈의 BPRLoss 클래스 타입으로 초기화
    bpr: utils.BPRLoss = loss_class

    # utils 모듈의 timer 객체 생성자 함수 호출
    with timer(name="Sample"):
        # S 변수에 담기는 정보가 무엇인지 모르겠음
        # numpy array 타입으로 반환되는 것 같음
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()  # 41830
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    # world 모듈의 device 변수와 config 딕셔너리 이용
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    # unobserved => negative의 의미를 담아 negItmes 변수를 생성한 것 같음
    # to는 _exceptions.py 또는 test_units.py 의 메소드
    # 하는 역할은 잘 모르겠음
    negItems = negItems.to(world.device)
    # 유저와 observed_items, unobserved_items를 utils 모듈의 shuffle 메소들 이용해 섞는 것 같음
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1  # 21
    aver_loss = 0.
    aver_mf_loss = 0.0
    aver_reg_loss = 0.0
    # utils 모듈의 minibatch 메소드를 이용해 제너레이터 타입으로 반환 받아서 enumerate 메소드의 인자로 전달
    for (batch_i,
         (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users,
                                                                           posItems,
                                                                           negItems,
                                                                           batch_size=world.config['bpr_batch_size'])):
        cri, mf_loss, reg_loss = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        aver_mf_loss += mf_loss
        aver_reg_loss += reg_loss
        # world 모듈의 tensorboard의 존재 유무 확인
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    aver_mf_loss = aver_mf_loss / total_batch
    aver_reg_loss = aver_reg_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss {aver_loss:.4f}  {aver_mf_loss:.4f}  {aver_reg_loss:.4f} - {time_info}"


# 배치 단위로 테스트 결과를 딕셔너리 타입으로 반환하는 메소드인 것 같음
# precision, recall, ndcg 값을 얻어올 때 utils 모듈의 RecallPrecision_ATk 메소드와 NDCGGatK_r 메소드를 이용
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    # world 모듈의 topks 변수만큼 for-loop를 순회
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
    
    # 자료형 힌트 제공
    # BasicDataset은 dataloader 모듈의 클래스로 raise 키워드가 많이 사용되었음
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    # valDict: dict = dataset.valDict

    # 추천 모델의 타입 힌트를 model 모듈의 GTN 클래스 타입으로 설정
    Recmodel: model.GTN
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    # with 구문: 자원 획득, 사용, 반납 프로세스를 한 번에 처리할 수 있도록 해줌
    # 객체의 라이프사이클을 설계할 수 있음
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
        # pre_results 리스트 생성
        # Procedure(현재) 모듈 내의 test_one_batch 메소드를 이용
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
        
        # world.tensorboard에 저장된 값이 0 이어서 아래 if문이 동작하고 있지 않음
        print(f'\nworld.tensorboard: {world.tensorboard}\n')

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
