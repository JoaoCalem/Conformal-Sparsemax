
from conformal_sparsemax.classifier import CNN, get_data, train, evaluate
from entmax.losses import SparsemaxLoss, Entmax15Loss
import json
import torch
from torch import nn
from sklearn.metrics import f1_score

loss = 'sparsemax' #sparsemax or softmax
dataset = 'MNIST' #CIFAR100 or MNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if loss == 'sparsemax':
    criterion = SparsemaxLoss()
elif loss == 'softmax':
    criterion = torch.nn.NLLLoss()
train_dataloader, dev_dataloader, test_dataloader, _ = get_data(0.2,16,dataset = dataset)

if dataset in ['CIFAR100', 'CIFAR10']:
    model = CNN(100,
                32,
                3,
                transformation=loss,
                conv_channels=[256,512,512],
                convs_per_pool=2,
                batch_norm=True,
                ffn_hidden_size=1024,
                kernel=5,
                padding=2).to(device)
elif dataset == 'MNIST':
    model = CNN(10,
                28,
                1,
                transformation=loss).to(device)
else:
    raise Exception('Wrong dataset name')
    
model, train_history, val_history, f1_history = train(model,
                                            train_dataloader,
                                            dev_dataloader,
                                            criterion,
                                            epochs=30,
                                            patience=30)

_, predicted_labels, true_labels, test_loss = evaluate(
                                                    model,
                                                    test_dataloader,
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