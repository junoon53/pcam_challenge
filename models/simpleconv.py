import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5, 1)
        self.conv2 = nn.Conv2d(50, 100, 5, 1)
        self.fc1 = nn.Linear(21*21*100, 1600)
        self.fc2 = nn.Linear(1600, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 21*21*100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        o = F.sigmoid(x)
        return o
