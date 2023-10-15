import numpy as np
import torch


min_range, max_range = -2., 2.
n = 20
step = (max_range - min_range) / float(n)

lines = []
    
for vl in np.arange(min_range, max_range, step):
    z = torch.stack([
        torch.FloatTensor([vl]*n),
        torch.FloatTensor([v2 for v2 in np.arange(min_range, max_range, step)]),            
        ], dim = -1)
    
print(z)