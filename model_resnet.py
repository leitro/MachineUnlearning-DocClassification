import torch
from torch import nn
from transformers import ResNetModel
import torch.nn.init as init

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-34').to(DEVICE)

        for m in self.resnet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.linear0 = nn.Linear(512, 256)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 64)
        self.proj = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img):
        ret = self.resnet(img)
        feat = ret.pooler_output.squeeze(-1).squeeze(-1) # batch, 512
        feat = self.relu(feat)
        feat = self.dropout(feat)
        feat = self.linear0(feat)
        feat = self.dropout(self.relu(feat))
        feat = self.linear1(feat)
        feat = self.dropout(self.relu(feat))
        feat = self.linear2(feat)
        feat2 = self.dropout(self.relu(feat))
        fin = self.proj(feat2)
        return feat, fin

