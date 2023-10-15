import torch

def load_mnist(is_train = True, flatten = True):
    from torchvision import datasets, transforms
    
    dataset = datasets.MNIST(
        '../data', train = is_train, download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    
    x = dataset.data.float() / 255.
    y = dataset.targets
    
    if flatten:
        x = x.view(x.size(0), -1)
        
    return x, y



def get_loaders(config):
    x, y = load_mnist(is_train = True, flatten = True)
    
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt
    
    # Shuffle dataset to split inot train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
        x,
        dim = 0,
        index = indices
    ).split([train_cnt, valid_cnt], dim = 0)
    
    train_y, valid_y = torch.index_select(
        y,
        dim = 0,
        index = indices
    ).split([train_cnt, valid_cnt], dim = 0)
        
    test_x, test_y = load_mnist(is_train = False, flatten = False)
    

    return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    
    
    
    
    
    
    
    
    
    
    