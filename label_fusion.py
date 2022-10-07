import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


class label_fusion(nn.Module):
    def __init__(self):
        super(label_fusion, self).__init__()
        # self.flatten = nn.Flatten(1, -1)
        self.r = nn.ReLU()
        self.dpt = nn.Dropout(p=0.5)   # to be tuned
        self.l1 = nn.Linear(60, 100)  # changed for single task here
        self.l2 = nn.Linear(100, 32)
        self.l3 = nn.Linear(32, 3)
        # self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        # n = x.size(0)
        # x = x.view(n, -1)
        # x = self.flatten(x)
        x = self.l1(x)
        x = self.r(x)
        x = self.dpt(x)  # to be tuned
        x = self.l2(x)
        x = self.r(x)
        x = self.l3(x)
        # x = self.sf(x)
        return x


# input = torch.rand(10,15,3)
# model = label_fusion()
# output = model(input)
# print(output)
# print(output.argmax(dim=1))




