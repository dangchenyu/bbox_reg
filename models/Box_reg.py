from models.res18 import Resnet18
import torch.nn.functional as F
import numpy
import torch
import torch.nn as nn


class Box_reg(nn.Module):
    def __init__(self):
        super(Box_reg, self).__init__()
        self.backbone = Resnet18()
        self.regresssion = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 4),
        )
    def forward(self, input):
        input=self.backbone(input)
        input=input.view(input.size(0),-1)
        output=self.regresssion(input)
        return output

if __name__ == '__main__':
    model=Box_reg()
    dumy=torch.rand(1,3,300,300)
    output=model(dumy)