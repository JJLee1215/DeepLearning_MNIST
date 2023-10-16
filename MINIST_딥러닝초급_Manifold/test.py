# 이 모델은 잘 이해하지 못했다.

import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_test
from model import Autoencoder

# 1) 저장한 모델 불러오기
# 2) data_loader로 부터 x_test, y_test 불러오기

def show_image(x):
    if x.dim == 1:
        x = x.view(int(x.size(0) ** .5), -1)
        
    plt.imshow(x, cmap = 'gray')
    plt.show()
    

checkpoint = torch.load('./checkpoint.pth') # checkpoint.keys() == model, config

model_state_dict = checkpoint['model']
model = Autoencoder()  # Instantiate your model class here
model.load_state_dict(model_state_dict)

test_x, test_y = get_test()
    
# Mean value in each space
with torch.no_grad():
    import random
    
    model.eval()
    
    index1 = int(random.random() * test_x.size(0))
    index2 = int(random.random() * test_x.size(0))
    
    z1 = model.encoder(test_x[index1].view(1,-1))
    z2 = model.encoder(test_x[index2].view(1,-1))
    
    recon = model.decoder((z1+z2)/2).view(28,-1)
    
    show_image(test_x[index1])
    show_image(test_x[index2])
    show_image((test_x[index1]+test_x[index2])/2)
    show_image(recon)

