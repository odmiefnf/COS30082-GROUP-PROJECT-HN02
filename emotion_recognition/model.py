"""Facial-expression CNNs on grayscale 1x48x48 inputs in [-1, 1]."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class EmotionCNN(nn.Module):
    """A small grayscale facial-expression CNN, initialized from scratch."""
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

class EmotionMobileNetV3(nn.Module):
    """ImageNet MobileNetV3-Large, partially fine-tuned, behind the same 1x48x48 contract.

    The grayscale face is resized to 112 px (a 4x4 final map instead of 2x2 at 48 px),
    repeated to three channels and ImageNet-normalized inside the model, so FER data
    loading, augmentation and app preprocessing are unchanged.
    """
    INPUT_SIZE=112
    def __init__(self,pretrained=False,freeze_blocks=4):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights,mobilenet_v3_large
        net=mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        net.classifier[3]=nn.Linear(net.classifier[3].in_features,7)
        for param in net.features[:freeze_blocks].parameters():
            param.requires_grad=False
        self.freeze_blocks=freeze_blocks
        self.net=net
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer("std",torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)
    def forward(self,images):
        if images.ndim!=4 or tuple(images.shape[1:])!=(1,48,48):
            raise ValueError("EmotionMobileNetV3 expects batches of grayscale 1x48x48 inputs.")
        images=F.interpolate(images,size=self.INPUT_SIZE,mode="bilinear",align_corners=False)
        return self.net(((images+1)/2-self.mean)/self.std)

ARCHITECTURES={"EmotionCNN-v1":EmotionCNN,"EmotionMobileNetV3-v1":EmotionMobileNetV3}
