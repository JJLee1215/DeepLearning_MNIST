from copy import deepcopy
import numpy as np

import torch

class Trainer():
    def __init__(self, model, optimizer, crit, config):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.config = config
    
    def _batchify(self, x, y, batch_size, random_split = True):
        '''
        input: x, y, and batch_size
        x와 y를 받아 random으로 자리를 바꾼 후, batch_size로 쪼갠다.
        '''
        if random_split:
            indice = torch.randperm(x.size(0), device = x.device) # 무작위 순열 인덱스는 텐서 x가 위치한 장치와 동일한 장치에서 생성
            x = torch.index_select(x, dim = 0, index = indice)
            y = torch.index_select(y, dim = 0, index = indice)
        
        x = x.split(batch_size, dim = 0)
        y = y.split(batch_size, dim = 0)
        
        return x, y    

    
    def _train(self, train_x, train_y, config):
        '''
        y_hat = model(train_x), train_y-y_hat의 오차가 최소가 되도록 W를 업데이트
        '''
        x, y = train_x, train_y
        x, y = self._batchify(x, y, config.batch_size)
        
        total_loss = 0
        
        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            
            loss_i = self.crit(y_hat_i, y_i.squeeze())
            
            self.optimizer.zero_grad()
            loss_i.backward()
            
            self.optimizer.step()
            
            if config.verbose >=2 :
                print("Train Iteration (%d/%d): loss = %.4e" %(i+1, len(x), float(loss_i)))
            
            total_loss += float(loss_i)
        
        return total_loss / len(x)
    
    
    def _validate(self, valid_x, valid_y, config):
        
        with torch.no_grad():
            x, y = valid_x, valid_y
            x, y = self._batchify(x, y, config.batch_size)
            
            total_loss = 0
            
            for i, (x_i, y_i) in enumerate(zip(x,y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())
                
                if config.verbose >=2 :
                    print("Train Iteration (%d/%d): loss = %.4e" %(i+1, len(x), float(loss_i)))
                
                total_loss += float(loss_i)
        
        return total_loss / len(x)
    
    
    def train(self, train_x, train_y, valid_x, valid_y, config):
        lowest_loss = np.inf
        best_model = None
        
        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_x, train_y, config)
            valid_loss = self._validate(valid_x, valid_y, config)
            
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
            
            print("Epoch (%d/%d): train_loss = %.4e   valid_loss = %4e   lowest_loss = %.4e" %(
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))
                
        self.model.load_state_dict(best_model)
        
        
        
        
        
        
        
        
        
        
        
        