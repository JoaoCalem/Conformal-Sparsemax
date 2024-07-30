from confpred import ConformalPredictor,SparseScore,SoftmaxScore
from confpred.classifier import CNN, evaluate
from confpred.datasets import CIFAR10, CIFAR100, MNIST

from entmax.losses import SparsemaxLoss
import torch

loss = 'softmax' #sparsemax, softmax or entmax15
transformation = 'logits'
dataset='CIFAR100' #CIFAR100 or MNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_class = 100 if dataset == 'CIFAR100' else 10
if dataset in ['CIFAR100','CIFAR10']:
    model = CNN(n_class,
                32,
                3,
                transformation=transformation,
                conv_channels=[256,512,512],
                convs_per_pool=2,
                batch_norm=True,
                ffn_hidden_size=1024,
                kernel=5,
                padding=2).to(device)
if dataset == 'MNIST':
    model = CNN(10,
                28,
                1,
                transformation=transformation).to(device)
    
data_class = {
    'CIFAR100': CIFAR100,
    'CIFAR10': CIFAR10,
    'MNIST': MNIST,
}

data = data_class[dataset](0.2, 16, 3000, True)

model.load_state_dict(torch.load(f'models/{dataset}_{loss}.pth'))
if loss == 'sparsemax':
    criterion = SparsemaxLoss()
elif loss == 'softmax':
    criterion = torch.nn.NLLLoss()
    
test_proba, _, test_true, _ = evaluate(
                                model,
                                data.test,
                                criterion,
                                True)

cal_proba, _, cal_true, _ = evaluate(
                                model,
                                data.cal,
                                criterion,
                                True)

cp = ConformalPredictor(SparseScore(2))
cp.calibrate(cal_true, cal_proba, 0.1)
print(cp.evaluate(test_proba, test_true))