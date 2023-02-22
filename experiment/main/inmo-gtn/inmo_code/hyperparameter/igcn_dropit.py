import sys

sys.path.append('/home1/prof/hwang1/seokhwi/gnn-recommender/experiment/main/inmo-gtn/inmo_code/')

local_python_path = '/home1/prof/hwang1/.local/lib/python3.8/site-packages'
if local_python_path in sys.path:
    sys.path.remove('/home1/prof/hwang1/.local/lib/python3.8/site-packages')

print()
for p in sys.path:
    print(p)
print()

from dataset import get_dataset
from model_inmo import get_model
from trainer import get_trainer
import torch
from utils_inmo import init_run, set_seed
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config


def main():
    log_path = __file__[:-3]
    set_seed(2021)
    # init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'] + 'dropit'

    temp = dataset_config['path']
    print(f'\n\n{temp}\n\n')

    # writer = SummaryWriter(log_path)
    # dataset = get_dataset(dataset_config)
    # model = get_model(model_config, dataset)
    # trainer = get_trainer(trainer_config, dataset, model)
    # trainer.train(verbose=True, writer=writer)
    # writer.close()

    dataset_config['path'] = dataset_config['path'][:-6]
    
    temp = dataset_config['path']
    print(f'\n\n{temp}\n\n')

    exit()

    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    trainer = get_trainer(trainer_config, new_dataset, model)
    results, _ = trainer.eval('test')
    print('Previous interactions test result. {:s}'.format(results))

    model.norm_adj = model.generate_graph(new_dataset)
    model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
    model.update_feat_mat()
    results, _ = trainer.eval('test')
    print('Updated interactions test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
