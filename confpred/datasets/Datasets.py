import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod

class Datasets(ABC):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            norm: bool = True
            ):

        train_dataset = self._get_dataset(norm, train=True)
        self._train_splits(train_dataset,
                           calibration_samples,valid_ratio, batch_size)

        test_dataset = self._get_dataset(norm, train=False)
        self._test = DataLoader(test_dataset, batch_size=batch_size)
    
    @property
    def train(self):
        return self._train
    
    @property
    def dev(self):
        return self._dev
    
    @property
    def cal(self):
        return self._cal
    
    @property
    def test(self):
        return self._test
    
    @abstractmethod
    def _dataset_class(self):
        pass

    def _dataset(self, norm):
        data_class, normalize = self._dataset_class()
        
        if norm:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        
        return data_class, transform
    
    def _get_dataset(self, norm, train=True):
        data_class, transform = self._dataset(norm)
        return data_class(
            root="data",
            train=train,
            download=True,
            transform=transform
        )

    def _train_splits(self, train_dataset,
                      calibration_samples, valid_ratio, batch_size):
        
        gen = torch.Generator()
        gen.manual_seed(0)
        
        train_dataset, cal_dataset = torch.utils.data.dataset.random_split(
            train_dataset, 
            [len(train_dataset)-calibration_samples, calibration_samples],
            generator=gen
        )

        nb_train = int((1.0 - valid_ratio) * len(train_dataset))
        nb_valid = int(valid_ratio * len(train_dataset))
        train_dataset, dev_dataset = torch.utils.data.dataset.random_split(
            train_dataset, [nb_train, nb_valid], generator=gen
        )
        
        self._train = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self._dev = DataLoader(dev_dataset, batch_size=batch_size)
        self._cal = DataLoader(cal_dataset, batch_size=batch_size)