"""A small grayscale facial-expression CNN, initialized from scratch."""
from __future__ import annotations
import torch
from torch import nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for inputs,outputs,dropout in ((1,16,.1),(16,32,.15),(32,64,.2)):
            layers += [nn.Conv2d(inputs,outputs,3,padding=1,bias=False),nn.BatchNorm2d(outputs),nn.ReLU(inplace=True),
                       nn.Conv2d(outputs,outputs,3,padding=1,bias=False),nn.BatchNorm2d(outputs),nn.ReLU(inplace=True),
                       nn.MaxPool2d(2),nn.Dropout2d(dropout)]
        self.features=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((3,3)),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64*3*3,128),nn.ReLU(inplace=True),nn.Dropout(.4),nn.Linear(128,7))
    def forward(self,images):
        if images.ndim!=4 or tuple(images.shape[1:])!=(1,48,48):
            raise ValueError("EmotionCNN expects batches of grayscale 1x48x48 inputs.")
        return self.head(self.features(images))
