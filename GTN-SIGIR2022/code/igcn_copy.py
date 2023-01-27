from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
# 패키지 경로 재설정
# from igcn_cf.dataset import get_dataset
# 기존 코드: from dataset import get_dataset ← 에러 표시 났었음
from dataset_copy import get_dataset
from utils_copy import set_seed, init_run
from model_copy import get_model
from trainer_copy import get_trainer


# 이 모듈이 INMO-LGCN인 것 같음
def fitness(lr, l2_reg, dropout, aux_reg):
    # 시드 설정을 왜 여기서도 하는지 모르겠음
    set_seed(2021)

    device = torch.device('cuda')
    
    # path 수정
    # dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
    #                   'device': device}

    dataset_config = {'name': 'ProcessedDataset', 'path': '../data/gowalla',
                        'device': device}

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': dropout, 'feature_ratio': 1.}

    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': l2_reg, 'aux_reg': aux_reg,
                        'device': device, 'n_epochs': 100, 'batch_size': 512, 'dataloader_num_workers': 6,
                        'test_batch_size': 100, 'topks': [20]}
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def main():
    # __file__은 현재 python file이 존재하는 경로를 반환
    # .py 지움
    log_path = __file__[:-3]

    # print(f'MODEL: {log_path}')
    # exit()

    # utils 모듈의 메소드
    # log 경로와 시드 값을 인자로 전달
    init_run(log_path, 2021)
    # 파라미터를 딕셔너리를 이용해 저장
    # learning rate, L2 regularization coefficient, dropout
    # aux_reg는 뭔지 모르겠음

    # param_grid = {'lr': [1.e-3], 'l2_reg': [0., 1.e-5], 'dropout': [0., 0.1, 0.3],
    #               'aux_reg': [1.e-3, 1.e-2, 1.e-1]}
    
    # 한 번만 돌도록 수정
    param_grid = {'lr': [1.e-3], 
                'l2_reg': [0.], 
                'dropout': [0.],
                'aux_reg': [1.e-3]}

    # sklearn 패키지의 하위 모듈(model_selection)의 클래스
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None

    for params in grid:
        # ndcg(평가 메트릭) 값이 가장 큰 파라미터를 찾음
        ndcg = fitness(params['lr'], params['l2_reg'], params['dropout'], params['aux_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))

# 라이브러리로 임포트될 때는 if 이하의 커맨드들이 실행되지 않음
if __name__ == '__main__':
    main()
