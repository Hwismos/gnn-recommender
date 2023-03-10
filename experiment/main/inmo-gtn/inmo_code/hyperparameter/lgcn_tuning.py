import sys

sys.path.append('/home1/prof/hwang1/seokhwi/gnn-recommender/experiment/main/inmo-gtn/inmo_code/')

local_python_path = '/home1/prof/hwang1/.local/lib/python3.8/site-packages'
if local_python_path in sys.path:
    sys.path.remove('/home1/prof/hwang1/.local/lib/python3.8/site-packages')

from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from utils_inmo import set_seed, init_run
from model_inmo import get_model
from trainer import get_trainer


def fitness(lr, l2_reg):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config = {'name': 'ProcessedDataset', 'path': './data/movie-len/',
                      'device': device}
    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': l2_reg,
                      'device': device, 'n_epochs': 1000, 'batch_size': 256, 'dataloader_num_workers': 6,
                      'test_batch_size': 30, 'topks': [20]}
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def main():
    log_path = __file__[:-3]
    # init_run(log_path, 2021)
    # param_grid = {'lr': [1.e-3], 'l2_reg': [0., 1.e-5, 1.e-4, 1.e-3]}
    param_grid = {'lr': [1.e-3], 'l2_reg': [0.]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness(params['lr'], params['l2_reg'])
        print('NDCG: {:.5f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.5f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))


if __name__ == '__main__':
    main()