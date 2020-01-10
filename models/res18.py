import torch
import torch.functional as F
import torch.nn as nn
import numpy as np


class Basicblock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, expansion=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel * expansion, 3, stride=stride, padding=1)
        self.batchmorm1 = nn.BatchNorm2d(output_channel * expansion)
        self.conv2 = nn.Conv2d(output_channel * expansion, output_channel * expansion, 3, padding=1)
        self.batchmorm2 = nn.BatchNorm2d(output_channel * expansion)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or input_channel != output_channel * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, output_channel * expansion,1, stride=stride, padding=0),
                nn.BatchNorm2d(output_channel * expansion))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input):
        res = self.shortcut(input)
        input = self.conv1(input)
        input = self.batchmorm1(input)
        input = self.relu(input)
        input = self.conv2(input)
        input = self.batchmorm2(input)
        input = input + res
        output = self.relu(input)
        return output


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.input_channel = 64
        self.layers = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1s = self._make_layer(64, 1, Basicblock, 2, 1)
        self.conv2s = self._make_layer(128, 1, Basicblock, 2, 2)
        self.conv3s = self._make_layer(256, 1, Basicblock, 2, 2)
        self.conv4s = self._make_layer(512, 1, Basicblock, 2, 2)
        self.max_pool = nn.MaxPool2d(3,3,1)
    def _make_layer(self, output_channel, expansion, block, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks-1)
        for stride in strides:
            layers.append(block(self.input_channel, output_channel, stride, expansion))
            self.input_channel = output_channel * expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        input = self.layers(input)
        input = self.conv1s(input)
        input = self.conv2s(input)
        input = self.conv3s(input)
        input = self.conv4s(input)
        output = self.max_pool(input)
        return output
if __name__ == '__main__':
    model=Resnet18()
    dumy=torch.rand(3,3,300,300)
    output=model(dumy)
