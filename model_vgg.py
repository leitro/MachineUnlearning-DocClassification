import torch
from torch import nn
from torchvision.models import vgg19_bn, vgg16_bn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.vgg = vgg19_bn(weights='IMAGENET1K_V1')
        self.vgg = vgg16_bn(weights='IMAGENET1K_V1')
        self.linear = nn.Linear(1000, 512)
        self.proj = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img):
        ret = self.vgg(img)
        feat = self.relu(ret)
        feat = self.dropout(feat)
        feat512 = self.linear(feat)
        fin = self.proj(feat512)
        return feat512, fin

