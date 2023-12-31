import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from data_loader import get_loaders

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required = True)
    p.add_argument('--gpu_id', type = int, default = 0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--train_ratio', type = float, default = .8)
    
    p.add_argument('--batch_size', type = int, default = 256)
    p.add_argument('--n_epochs', type = int, default = 20)
    
    p.add_argument('--n_layers', type = int, default = 5)
    p.add_argument('--use_dropout', action = 'store_true')
    p.add_argument('--dropout_p', type = float, default = .3)
    
    p.add_argument('--verbose', type = int, default = 2)
    
    config = p.parse_args()
    
    return config

def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_loaders(config)
    
    print("train_x, train_y: ", train_x.shape, train_y.shape)
    print("valid_x, valid_y: ", valid_x.shape, valid_x.shape)      
        
    model = ImageClassifier(28**2, 10).to(device)
    optimizer = optim.Adam(model.parameters()) # parameters가 무엇인지 확인하기
    crit = nn.NLLLoss()
    
    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)
    
    
    trainer = Trainer(model, optimizer, crit, config)
    trainer.train(train_x, train_y, valid_x, valid_y, config)
    
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)
    
