
from confpred.classifier import CNN, train, evaluate
from confpred.datasets import CIFAR10, CIFAR100, MNIST
from entmax.losses import SparsemaxLoss, Entmax15Loss
import json
import torch
from torch import nn
from sklearn.metrics import f1_score

loss = 'sparsemax' #sparsemax or softmax
dataset = 'CIFAR10' #CIFARx =100 or MNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if loss == 'sparsemax':
    criterion = SparsemaxLoss()
elif loss == 'softmax':
    criterion = torch.nn.NLLLoss()

data_class = {
    'CIFAR100': CIFAR100,
    'CIFAR10': CIFAR10,
    'MNIST': MNIST,
}

data = data_class[dataset](0.2, 16, 3000, True)


n_class = 100 if dataset == 'CIFAR100' else 10
if dataset in ['CIFAR100','CIFAR10']:
    model = CNN(n_class,
                32,
                3,
                transformation=loss,
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
                transformation=loss).to(device)
    
model, train_history, val_history, f1_history = train(model,
                                            data.train,
                                            data.dev,
                                            criterion,
                                            epochs=25,
                                            patience=3)

_, predicted_labels, true_labels, test_loss = evaluate(
                                                    model,
                                                    data.test,
                                                    criterion)

f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f'Test loss: {test_loss:.3f}')
print(f'Test f1: {f1:.3f}')

results = {
    'train_history':train_history,
    'val_history':val_history,
    'f1_history':f1_history,
}

with open(f'results/{dataset}_{loss}_results.json', 'w') as f:
    json.dump(results, f)
    
torch.save(model.state_dict(), f'models/{dataset}_{loss}.pth')