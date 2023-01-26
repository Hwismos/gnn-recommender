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

# 추천 모델 객체 생성 → GTN
# Recmodel = register.MODELS[world.model_name](world.config, dataset, world.args)
# LightGCN
# Recmodel = register.MODELS[world.model_name](world.config, dataset)

# print(f'DEVICE: {world.device}')
# exit()

if world.model_name == 'gtn':
    Recmodel = register.MODELS[world.model_name](world.config, dataset, world.args)
else:
    Recmodel = register.MODELS[world.model_name](world.config, dataset)

cprint(world.model_name)

Recmodel = Recmodel.to(world.device)


# utils 모듈을 이용해 BPRLoss 설정
# 인자로 전달하는 추천 모델이 PairWiseModel
# world 모듈의 config 값도 인자로 전달
bpr = utils.BPRLoss(Recmodel, world.config)

# 데이터 디렉토리의 경로 설정
baisc_path_log = "../data/" + str(world.args.dataset) + "/log/"

# utils 모듈의 getFileName 메소들 호출
weight_file = utils.getFileName()
# 가중치 파일의 저장 경로 출력

# 로딩이 성공했을 때 try-catch문 실행
# 기존에 학습시킨 모델이 있는 경우인 것 같음
if world.LOAD:
    try:
        # load_state_dict 메소드가 어디 있는지 모르겠음
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        # word 모듈의 cprint 메소드를 이용해 모델 가중치가 로딩된 경로를 출력
        world.cprint(f"loaded model weights from {weight_file}")
    # 에러 처리
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
# 새로운 모델을 학습
try:
    # time 모듈을 이용해 timeStamp 변수 초기화
    timeStamp = time.time()

    # arguments 출력
    print(world.args)
    import matplotlib.pyplot as plt

    y = []
    # 학습 epochs 만큼의 원소를 갖는 리스트 생성
    x = list(range(world.TRAIN_epochs))
    # figure 메소드: 새로운 figure를 생성하거나 존재하는 figure를 활성화 시킴
    plt.figure()
    # epochs 수만큼 for-loop
    for epoch in range(world.TRAIN_epochs):
        # 시작 시간 업데이트
        start = time.time()

        # Procedure 모듈의 BPR_train_original 객체 생성
        # 에포크마다 객체를 다시 생성?
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)

        if epoch % 5 == 0:
            cprint("[TEST]")
            # 에포크가 5의 배수일 때마다 Procedure 모듈의 Test 객체 생성
            # 테스트 데이터 결과를 results 변수에 저장
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)

            # results 딕셔너리를 분해해서 precision, recall, ndcg 값을 각각 저장
            pre = round(results['precision'][0], 5)
            recall = round(results['recall'][0], 5)
            ndcg = round(results['ndcg'][0], 5)

            # top k 텍스트 생성 및 출력
            topk_txt = f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}]  {output_information} | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
            print(topk_txt)
    
        # 종료 시간 업데이트
        end = time.time()
        # 학습에 걸린 시간 계산
        diff = datetime.datetime.fromtimestamp((end - start))
        diff_mins = diff.strftime("%M:%S")
        print(
            f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}  |  {diff_mins}mins | Results val Top-k (recall, ndcg):  {recall}, {ndcg}')

# try 이후에 에러가 발생하더라도 무조건 실행
finally:
    if world.tensorboard:
        # None모듈에서 close 메소드를 호출하는 이유를 모르겠음
        w.close()
