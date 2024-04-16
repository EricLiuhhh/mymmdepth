import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mmdepth.registry import MODELS

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

@MODELS.register_module()
class S2DUpBlock(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels):
        super(S2DUpBlock, self).__init__()
        self.unpool = Unpool(in_channels)
        self.upper_branch = nn.Sequential(OrderedDict([
            ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
            ('batchnorm1', nn.BatchNorm2d(out_channels)),
            ('relu',      nn.ReLU()),
            ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
            ('batchnorm2', nn.BatchNorm2d(out_channels)),
        ]))
        self.bottom_branch = nn.Sequential(OrderedDict([
            ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.upper_branch(x)
        x2 = self.bottom_branch(x)
        x = x1 + x2
        x = self.relu(x)
        return x