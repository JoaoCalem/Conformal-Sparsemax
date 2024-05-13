import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import sparsemax, entmax15

class CNN(nn.Module):
    def __init__(
                self,
                transformation='softmax',
                n_classes=100,
                input_size=32,
                channels=3,
                kernel=5,
                padding=0
            ):
        super().__init__() 
        self.conv1 = nn.Conv2d(channels, 8, kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel)
        size_adjust = 2*padding-kernel+1
        self.fc1 = nn.Linear(16 * (size_adjust+(input_size+size_adjust)//2)**2, 512)
        self.fc2 = nn.Linear(512, n_classes)
        if transformation=='softmax':
            self.forward = self.softmax
        elif transformation=='sparsemax':
            self.forward = self.sparsemax
        else:
            raise Exception("Parameter 'transformation' must be 'softmax' or 'sparsemax'")

    
    def softmax(self,x):
        x = self.logits(x)
        x = nn.Softmax(-1)(x)
        return x
    
    def sparsemax(self,x):
        x = self.logits(x)
        x = sparsemax(x,-1)
        return x
    
    def logits(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN_CIFAR(nn.Module):
    def __init__(
                self,
                transformation='softmax',
                conv_channels=[256,512,512],
                convs_per_pool=2,
                batch_norm=True,
                ffn_hidden_size=1024,
                n_classes=100,
                input_size=32,
                channels=3,
                kernel=5,
                padding=0
            ):
        super().__init__() 
        self.convs_per_pool = convs_per_pool
        size_adjust = 2*padding-kernel+1
        channel_previous = channels
        self.convs = nn.ModuleList([])
        self.batch_norms=nn.ModuleList([])
        for channels in conv_channels:
            for i in range(convs_per_pool):
                self.convs.append(nn.Conv2d(channel_previous,
                                            channels,
                                            kernel,padding="same"
                                            ))
                if batch_norm:
                    self.batch_norms.append(nn.BatchNorm2d(channels))
                channel_previous=channels
            
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(8192, ffn_hidden_size)
        self.b1d = None
        if batch_norm:
            self.b1d = nn.BatchNorm1d(ffn_hidden_size,0.005,0.95)
        self.fc2 = nn.Linear(ffn_hidden_size, n_classes)
        
        self.transformation = transformation
        if transformation=='softmax':
            self.final = lambda x: nn.LogSoftmax(-1)(x)
        elif transformation=='sparsemax':
            self.final = lambda x: sparsemax(x,-1)
        else:
            raise Exception("Parameter 'transformation' must be 'softmax' or 'sparsemax'")
    
    def forward(self,x):
        for i in range(0,len(self.convs),self.convs_per_pool):
            for j in range(self.convs_per_pool):
                if self.batch_norms:
                    x = F.relu(self.batch_norms[i+j](self.convs[i+j](x)))
                else:
                    x = F.relu(self.convs[i+j](x))
            x = self.dropout(self.pool(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        if self.b1d:
            x = self.b1d(x)
        x = self.fc2(x)
        x = self.final(x)
        return x
    
    def eval(self):
        super().eval()
        if self.transformation=='softmax':
            self.final = lambda x: nn.Softmax(-1)(x)
    
    def train(self, mode=True):
        super().train(mode)
        if self.transformation=='softmax':
            self.final = lambda x: nn.LogSoftmax(-1)(x)
        