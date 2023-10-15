# 목표: MNIST 데이터를 입력받는 auto encoder model을 만든다.
# 1. MNIST 데이터 차원이 들어간다. R^28*28 공간
# 2. bottle neck 구간을 만든다. R^10 공간
# 3. MNIST 데이터가 빠져 나간다. R^28*28 공간
# 28*28 -> 500 // ... // 10 -> bottle_neck_size  <== btl_size는 10 이하가 되야 한다.

import torch.nn as nn 

class Autoencoder(nn.Module):
    def __init__(self, btl_size = 2):
        self.btl_size = btl_size
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            
            nn.Linear(10, btl_size),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            
            nn.Linear(500, 28*28),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        
        return y