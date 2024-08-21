from confpred.datasets import Datasets

import torchvision
import torchvision.transforms as transforms

class MNIST(Datasets):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            norm: bool = True
            ):
        super().__init__(valid_ratio, batch_size, calibration_samples, norm)
    
    def _dataset_class(self):
        data_class = torchvision.datasets.EMNIST
        normalize = transforms.Normalize(0.5, 0.5)
        return data_class, normalize
    
    def _get_dataset(self, norm, train=True):
        data_class, transform = self._dataset(norm)
        return data_class(
            root="data",
            split='byclass',
            train=train,
            download=True,
            transform=transform
        )