from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset_copy import get_dataset
from utils_copy import set_seed, init_run
from model_copy import get_model
from trainer_copy import get_trainer


def fitness(lr, l2_reg):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config = {'name': 'ProcessedDataset', 'path': '../data/gowalla',
                      'device': device}
    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    # epochs 1000 → 5로 줄여서 실행
    # gtn과 igtn을 만들어야 함
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': l2_reg,
                      'device': device, 'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'lr': [1.e-3], 'l2_reg': [0.]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness(params['lr'], params['l2_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))


if __name__ == '__main__':
    main()
