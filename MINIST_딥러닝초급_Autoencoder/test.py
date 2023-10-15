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

with torch.no_grad():
    import random
    
    index = int(random.random() * test_x.size(0))
    
    model.eval()
    
    print("test_x[index].size()           : ",test_x[index].size())
    print("test_x[index].view(1,-1).size(): ", test_x[index].view(1,-1).size())
    
    recon = model(test_x[index].view(1,-1))
    print("recon.size(): ",recon.size())
         
    recon = recon.view(28,-1)
    print("recon.view(28,-1)              : ", recon.size())
    
    show_image(test_x[index])
    show_image(recon.view(28,-1))
    
# R^2 공간 (latent space)에 있는 좌표를 시각화 해보기

color_map = ['brown', 'red', 'orange', 'yellow', 'green', 'blue', 'navy', 'purple', 'gray', 'black',]

plt.figure(figsize = (20,10))

with torch.no_grad():
    
    test_x_resize = test_x.view(test_x.size(0), -1)
    test_y_resize = test_y.view(test_y.size(0), -1)
    
    print("test_x_resize: {}, test_y_resize: {}".format(test_x_resize.size(), test_y_resize.size()))
    
    latents = model.encoder(test_x_resize[:1000])
    # figure를 z 2차원으로 줄인 결과 값.
    print("latents:", latents.size())
            
    for i in range(10):
        target_latents, target_y = [], []
        target_latents = [latents[j] for j, test_y_j in enumerate(test_y_resize[:1000]) if test_y_j == i]
        target_y = [test_y[j] for j, test_y_j in enumerate(test_y_resize[:1000]) if test_y_j == i]    
        
        target_latents = torch.stack(target_latents)
        target_y = torch.tensor(target_y)
        
        plt.scatter(target_latents[:, 0],
                    target_latents[:, 1],
                    marker = 'o',
                    color = color_map[i],
                    label = i)
        
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()


# 2차원에 표시된 숫자들이 어떻게 시각적으로 보이는지 확인해보기 // 이 부분은 다시한번 공부가 필요함
min_range, max_range = -2., 2.
n = 20
step = (max_range - min_range) / float(n)

with torch.no_grad():
    lines = []
    
    for v1 in np.arange(min_range, max_range, step):
        z = torch.stack([
            torch.FloatTensor([v1]*n),
            torch.FloatTensor([v2 for v2 in np.arange(min_range, max_range, step)]),            
        ], dim = -1)
        
        line = torch.clamp(model.decoder(z).view(n, 28, 28), 0, 1)
        line = torch.cat([line[i] for i in range(n -1, 0, -1)], dim = 0)
        lines += [line]
        
    lines = torch.cat(lines, dim = -1)
    plt.figure(figsize = (20,20))
    show_image(lines)     