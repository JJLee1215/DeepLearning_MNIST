import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import Autoencoder
from trainer import Trainer
from data_loader import get_loaders

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--gpu_id', type = int, default = 0 if torch.cuda.is_available() else - 1)
    
    p.add_argument('--train_ratio', type = float, default = .8)
    
    p.add_argument('--batch_size', type = int, default = 256)
    p.add_argument('--n_epochs', type = int, default =20)
    
    p.add_argument('--verbose', type = int, default = 1)
    
    p.add_argument('--btl_size', type = int, default = 2)
    
    config = p.parse_args()
    
    return config

def main(config):
    
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_loaders(config)
    
    print("train_x, train_y: ", train_x.shape, train_y.shape)
    print("valid_x, valid_y: ", valid_x.shape, valid_y.shape)
    
    model = Autoencoder(2).to(device)
    optimizer = optim.Adam(model.parameters()) # parameters가 무엇인지 확인하기
    crit = nn.MSELoss()
    
    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)
    
    
    trainer = Trainer(model, optimizer, crit, config)
    trainer.train(train_x, train_x, valid_x, valid_x, config) # <= 이 부분은 y값을 쓰지 않는 다는 것에 주의 하자
    
    torch.save({
        'model' : trainer.model.state_dict(),
        'config' : config,
    },'checkpoint.pth')


if __name__ == '__main__':
    config = define_argparser()
    main(config)