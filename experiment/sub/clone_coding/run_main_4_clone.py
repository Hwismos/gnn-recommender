import world
from world import cprint
import utils
import Procedure
import register
from register import dataset

import torch
import numpy as np
from tensorboardX import SummaryWriter
import time, datetime
from os.path import join

import random
import numpy as np

seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# ==================kill============
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==================================

# 이 모듈은 클래스나 메소드가 없이 순차적으로 실행됨

# register 모듈의 딕셔너리를 이용해 바로 객체를 생성
Recmodel=register.MODELS[world.model_name](world.config,
                                            dataset,
                                            world.args)
# converesion
Recmodel = Recmodel.to(world.device)

# BPRLoss 객체 생성
bpr=utils.BPRLoss(Recmodel, world.config)

baisc_path_log='../data/' + str(world.args.dataset) + '/log'        # ? 안 쓰임

# 저장된 가중치 파일이 존재한다면 load
# 로딩이 된 경우가 파일을 찾을 수 없는 경우에 대한 예외처리
weight_file=utils.getFileName()
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k=1     # ? 안 쓰임

# tensorboard 확인해서 있으면 SummaryWriter 객체 생성
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

final_topk_txt=""       # ? 안 쓰임

try:
    timeStamp = time.time()
    print(world.args)
    
    import matplotlib.pyplot as plt

    y=[]        # ? 안 쓰임
    x=list(range(world.TRAIN_epochs))   # [0, 1, ... 99]
    plt.figure()        # 막상 그림은 안 나옴

    for epoch in range(world.TRAIN_epochs):
        start=time.time()

        # 딕셔너리 타입으로 한 epoch에 대한 학습 정보를 저장
        output_information=Procedure.BPR_train_original(dataset,
                                                        Recmodel,
                                                        bpr,
                                                        epoch,
                                                        neg_k=Neg_k,
                                                        w=w)

        if epoch % 5 == 0:
            cprint('[TEST]')
            
            # w를 인자로 받는거보면 저장된 weight_file을 이용?
                # add_scalar 메소드를 호출
                # default는 None
            results=Procedure.Test(dataset, 
                                    Recmodel,
                                    epoch,
                                    w,
                                    world.config['multicore'],
                                    val=False)
            pre=round(results['precision'][0], 5)
            recall=round(results['recall'][0], 5)
            ndcg=round(results['ndcg'][0], 5)

            topk_txt=f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}]  \
                        {output_information} \
                        | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
            print(topk_txt)     # final_topk_txt가 원래 이쯤에서 쓰였어야 했나봄

        end=time.time()
        diff=datetime.datetime.fromtimestamp((end-start))
        diff_mins=diff.strftime("%M%S")
        print(
            f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] \
            {output_information}  \
            |  {diff_mins}mins \
            | Results val Top-k (recall, ndcg):  {recall}, {ndcg}')
finally:
    # tensorboard가 켜져있으면 파일을 close
    if world.tensorboard:
        w.close()