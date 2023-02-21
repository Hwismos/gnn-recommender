import sys

sys.path.append('/home1/prof/hwang1/seokhwi/gnn-recommender/experiment/main/igcn_cf/')
sys.path.remove('/home1/prof/hwang1/.local/lib/python3.8/site-packages')

for path in sys.path:
    print(path)
print(f'\n 모듈 checking...\n')

from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config

print('한 번 더 확인')
exit()

def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')

    print(f'DEVICE: {device}')
    exit()

    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + str(1)

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()
    results, _ = trainer.eval('test')


    print('Test result. {:s}'.format(results))

# python -u -m run.run
# run 패키지 아래 있는 run.py 모듈을 실행하는 스크립트
if __name__ == '__main__':
    # main()
    exit()
