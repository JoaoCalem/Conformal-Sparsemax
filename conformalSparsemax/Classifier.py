import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, loss='softmax', n_classes=10, input_size=28, channels=1, kernel=5, padding=0):
        super().__init__() 
        size_adjust = 2*padding-kernel+1
        self.conv1 = nn.Conv2d(channels, 8, kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel)
        self.fc1 = nn.Linear(16 * (size_adjust+(input_size+size_adjust)//2)**2, 512)
        self.fc2 = nn.Linear(512, n_classes)
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x