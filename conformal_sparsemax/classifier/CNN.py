import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import sparsemax, entmax15

class CNN(nn.Module):
    def __init__(self, loss='softmax', n_classes=100, input_size=32, channels=3, kernel=5, padding=0):
        super().__init__() 
        size_adjust = 2*padding-kernel+1
        self.conv1 = nn.Conv2d(channels, 8, kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel)
        self.fc1 = nn.Linear(16 * (size_adjust+(input_size+size_adjust)//2)**2, 512)
        self.fc2 = nn.Linear(512, n_classes)
        if loss=='softmax':
            self.final = lambda x: nn.Softmax(-1)(x)
        elif loss=='sparsemax':
            self.final = lambda x: sparsemax(x,-1)
        elif loss=='entmax15':
            self.final = lambda x: entmax15(x,-1)
        else:
            raise Exception("Parameter 'loss' must be 'softmax' or 'sparsemax'")
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final(x)
        return x
    
class CNN_CIFAR(nn.Module):
    def __init__(self, loss='softmax', n_classes=100, input_size=32, channels=3, kernel=5, padding=0):
        super().__init__() 
        size_adjust = 2*padding-kernel+1
        self.conv1 = nn.Conv2d(channels, 32, kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel)
        self.conv3 = nn.Conv2d(64, 64, kernel)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, n_classes)
        if loss=='softmax':
            self.final = lambda x: nn.Softmax(-1)(x)
        elif loss=='sparsemax':
            self.final = lambda x: sparsemax(x,-1)
        elif loss=='entmax15':
            self.final = lambda x: entmax15(x,-1)
        else:
            raise Exception("Parameter 'loss' must be 'softmax' or 'sparsemax'")
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final(x)
        return x