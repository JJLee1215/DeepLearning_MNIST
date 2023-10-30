'''
정리해야할 개념
batchNorm1d 2d 차이
x = x.view(-1, 1, x.size(-2), x.size(-1))
MNIST tensor가 CNN 안에서 어떻게 흘러 가는지 파악하기
'''


import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3,3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3,3), stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        # |x| = (batch_size, in_channels, h, w)
        
        y = self.layers(x)
        
        # |y| = (batch_size, out_channels, h, w)
        
        return y
    
class ConvlutionalClassifier(nn.Module):
    def __init__(self, output_size):
        self.output_size = output_size
        
        super().__init__()
        
        self.blocks = nn.Sequential(
            ConvolutionalBlock(1, 32),
            ConvolutionalBlock(32,64),
            ConvolutionalBlock(64,128),
            ConvolutionalBlock(128,256),
            ConvolutionalBlock(256,512),
        )
        
        self.layers = nn.Sequential(
            nn.Linear(512,50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.Softmax(dim = -1),
        )
        
    def forward(self, x):
        assert x.dim() > 2
        
        if x.dim() == 3:
            # |x| = (batch_size, h, w)
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        # |x| = (batch_size, 1, h, w)
        
        z = self.block(x)
        # |z| = (batch_size, 512, 1, 1)
        
        y = self.layers(z.squeeze())
        # |y| = (batch_size, output_size)
        
        return y
    