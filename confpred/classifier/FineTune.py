"""Joao Calem - CNN.py"""
# Load model directly
from transformers import AutoModelForImageClassification

import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import sparsemax, entmax15

from typing import List

class FineTuneViT(nn.Module):
    def __init__(self,
            n_classes: int,
            transformation: str = 'softmax',
            ):
        """
        Constructor for CNN model
        """
        
        super().__init__() 
        
        self._transformation = transformation

        self.vit = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.classifier = nn.Linear(self.vit.classifier.in_features, n_classes)
        
        self.train()
    
    def forward(self, x):
        """
        Forward pass for specified transformation function on intitialisation.
        """
        
        x = self.vit(x)[0]
        
        return self._final(x)

    
    def eval(self):
        """
        Set model to evaluation mode.
        """
        
        super().eval()
        if self._transformation=='softmax':
            self._final = lambda x: nn.Softmax(-1)(x)
        elif self._transformation=='sparsemax':
            self._final = lambda x: sparsemax(x,-1)
        elif self._transformation=='entmax':
            self._final = lambda x: entmax15(x,-1)
    def train(self, mode=True):
        """
        Set model to training mode.
        """
        
        super().train(mode)
        if self._transformation=='softmax':
            self._final = lambda x: nn.LogSoftmax(-1)(x)
        #elif self._transformation in ['logits', 'sparsemax']:
        else:
            self._final = lambda x: x
        #else:
        #    raise Exception(
        #        "Parameter 'transformation' must be 'softmax', 'sparsemax' ot 'logits"
        #        )
        
