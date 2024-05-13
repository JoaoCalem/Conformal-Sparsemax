import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data(valid_ratio, batch_size, calibration_samples=3000, dataset='CIFAR100'):
    
    if dataset=='CIFAR100':
        data_class = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(0.5, 0.5, 0.5)
        
    elif dataset=='CIFAR10':
        data_class = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize(0.5, 0.5, 0.5)
        
    elif dataset=='MNIST':
        data_class = torchvision.datasets.MNIST
        normalize = transforms.Normalize(0.5, 0.5)
    else:
        raise Exception("Variable 'dataset' must be 'CIFAR100' or 'MNIST'")
        
    transform = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    train_valid_dataset = data_class(
        root="data",
        train=True,
        download=True,
        transform=transform
    )


    test_dataset = data_class(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    
    gen = torch.Generator()
    gen.manual_seed(0)
    
    train_valid_dataset, cal_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, 
        [len(train_valid_dataset)-calibration_samples, calibration_samples],
        generator=gen
    )
    
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    train_dataset, dev_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, [nb_train, nb_valid], generator=gen
    )
    
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    cal_dataloader = DataLoader(cal_dataset, batch_size=batch_size)
    
    return train_dataloader, dev_dataloader, test_dataloader, cal_dataloader