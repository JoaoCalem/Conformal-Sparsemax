import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data(valid_ratio, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)])

    train_valid_dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )


    test_dataset = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    train_dataset, dev_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, [nb_train, nb_valid]
    )
    
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, dev_dataloader, test_dataloader