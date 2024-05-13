from conformal_sparsemax.classifier import CNN, get_data, evaluate
from sklearn.metrics import f1_score,accuracy_score
import torch
import numpy as np
import pickle
from entmax.losses import SparsemaxLoss, Entmax15Loss

loss = 'sparsemax' #sparsemax, softmax or entmax15
transformation = 'softmax'
dataset='CIFAR10' #CIFAR100 or MNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if dataset == 'CIFAR100':
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
elif dataset == 'CIFAR10':
    model = CNN(10,
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

_,_, test_dataloader, cal_dataloader = get_data(0.2,16,dataset = dataset)
model.load_state_dict(torch.load(f'models/{dataset}_{loss}.pth'))
if loss == 'sparsemax':
    criterion = SparsemaxLoss()
elif loss == 'entmax15':
    criterion = Entmax15Loss()
elif loss == 'softmax':
    criterion = torch.nn.NLLLoss()
test_proba, test_pred, test_true, test_loss = evaluate(
                                                    model,
                                                    test_dataloader,
                                                    criterion)

test_f1 = f1_score(test_true, test_pred, average='weighted')
test_acc = accuracy_score(test_true, test_pred)

print(f'Test loss: {test_loss:.3f}')
print(f'Test f1: {test_f1:.3f}')
print(f'Test Accuracy: {test_acc:.3f}')

cal_proba, cal_pred, cal_true, cal_loss = evaluate(
                                                    model,
                                                    cal_dataloader,
                                                    criterion)

cal_f1 = f1_score(cal_true, cal_pred, average='weighted')
cal_acc = accuracy_score(cal_true, cal_pred)

print(f'Calibration loss: {cal_loss:.3f}')
print(f'Calibration f1: {cal_f1:.3f}')
print(f'Calibration acc: {cal_acc:.3f}')

#One Hot Encoding
test_true_enc = np.zeros((test_true.size, test_true.max()+1), dtype=int)
test_true_enc[np.arange(test_true.size),test_true] = 1

cal_true_enc = np.zeros((cal_true.size, cal_true.max()+1), dtype=int)
cal_true_enc[np.arange(cal_true.size),cal_true] = 1



predictions = {'test':{'proba':test_proba,'true':test_true_enc},
 'cal':{'proba':cal_proba,'true':cal_true_enc}}

loss = "NLLLoss" if loss=="softmax" else "FYLoss"
for dataset_type in ['cal','test']:
    with open(f'predictions/{dataset}_{dataset_type}_true.pickle', 'wb') as f:
        pickle.dump(predictions[dataset_type]['true'], f)
    with open(
        f'predictions/{dataset}_{dataset_type}_{loss}' +
            f'_{transformation}_{"proba"}.pickle'
        , 'wb'
    ) as f:
        pickle.dump(predictions[dataset_type]["proba"], f)