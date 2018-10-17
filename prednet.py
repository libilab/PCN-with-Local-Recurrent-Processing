'''PredNet in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class features2(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=7, stride=2, padding=3, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.featBN = nn.BatchNorm2d(outchan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.featBN(self.conv(x)))
        return y

class PcConvRes(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, cls=0, bias=False):
        super().__init__()
        self.FFconv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, kernel_size, stride, padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,outchan,1,1))])
        self.relu = nn.ReLU(inplace=True)
        self.cls = cls
        self.shortcut = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        y = self.relu(self.FFconv(x))
        b0 = F.relu(self.b0[0]+1.0).expand_as(y)
        for _ in range(self.cls):
            y = self.FFconv(self.relu(x - self.FBconv(y)))*b0 + y
        y = y + self.shortcut(x)
        return y

''' Architecture PredNetResA '''
class PredNetResD(nn.Module):
    def __init__(self, num_classes=1000, cls=0, Tied = False):
        super().__init__()
        # self.ics =     [    3,   64,   64,  128,  128,  128,  256,  256,  256,  256,  512,  512] # input chanels
        # self.ocs =     [   64,   64,  128,  128,  128,  256,  256,  256,  256,  512,  512,  512] # output chanels
        # self.maxpool = [False, True, True,False,False, True,False,False,False, True,False,False] # downsample flag
        self.ics =     [    3,   64,   64,  128,  128,  128,  128,  256,  256,  256,  512,  512] # input chanels
        self.ocs =     [   64,   64,  128,  128,  128,  128,  256,  256,  256,  512,  512,  512] # output chanels
        self.maxpool = [False,False, True,False, True,False, True,False,False, True,False,False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)

        self.baseconv = features2(self.ics[0], self.ocs[0])
        # construct PC layers
        if Tied == False:
            self.PcConvs = nn.ModuleList([PcConvRes(self.ics[i], self.ocs[i], cls=self.cls) for i in range(1, self.nlays)])
        else:
            self.PcConvs = nn.ModuleList([PcConvResTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(1, self.nlays)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(1, self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        x = self.baseconv(x)
        for i in range(self.nlays-1):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = self.relu(self.BNend(x))
        out = F.avg_pool2d(out, kernel_size=7, stride=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

