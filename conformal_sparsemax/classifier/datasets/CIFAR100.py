from conformal_sparsemax.classifier.datasets import Datasets

import torchvision
import torchvision.transforms as transforms

class CIFAR100(Datasets):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            norm: bool = True
            ):
        super().__init__(valid_ratio, batch_size, calibration_samples, norm)
    
    def _dataset_class(self):
        data_class = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(0.5, 0.5, 0.5)
        return data_class, normalize