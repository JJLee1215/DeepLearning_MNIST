'''
Purpose : building fully connected layers
input size : 28*28, 500
500 -> 400 -> 300 -> 200 -> 100 -> 50 -> output 

1. BatchNorm - normalize for mini-batches 

Regularization = minimize training error 
1) Weight Decay
2) Dropout
=> training time is delayed


input tensor shape = |X| = (batch_size, vector_size)
mu = x.mean(dim = 0), sigma = x.std (dim = 0)
|mu| = |sigma| = (vs,) <- the same dimension as vector_size

-> Unit Gaussaian 

    y = gamma (x - mu)/(sigma^2 + e)^0.5 + beta

    gamma, beta <- Learning parameters updated by backpropagation!!

"How much should we increase and shift to find favorable parameters for learning?"
 * gamma -> increase
 * shift -> beta

first step: regularization
second step: applying gamma and shift

Caution!

Learning: calculate mu and sigma within Mini-batches.
Inference: calculate mu and sigma within Mini-batches -> cheating => calculate the average and std from the accumulated input.
'''

import torch
import torch.nn as nn

class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        super.__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.logsoftmax(dim = -1)
        )
        
    def forward(self,x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)
        
        # |y} = (batch_size, output_size)
        
        return y
    
    
    
    
    