import math
from typing import List, Union
from scipy.stats import truncnorm
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
from .base_guided_net import BaseGuidedNet
    
class StoDepthBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, m, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepthBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = m
        self.multFlag = multFlag
        self.out_channels = planes * self.expansion

    def forward(self, x):

        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out
    
@MODELS.register_module()
class StoDepthResNet(BaseGuidedNet):

    def __init__(self,
                 in_channels=3,
                 blocktype='BasicBlock',
                 guides=None,
                 guide_locations=None,
                 hook_positions={'l':(1, 2, 3, 4, 5)},
                 drop_prob=0.5,
                 base_channels=32,
                 stem_channels=None,
                 stem_norm=True,
                 multFlag=True, 
                 num_blocks=(2, 2, 2, 2, 2),
                 plane_ratios=(2, 4, 8, 8, 8),
                 strides=(1, 2, 2, 2, 2),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(guides=guides, guide_locations=guide_locations, hook_positions=hook_positions, init_cfg=init_cfg)
        assert len(num_blocks) == len(plane_ratios) == len(strides)
        if stem_channels is None:
            stem_channels = base_channels
        if blocktype == 'BasicBlock':
            block = StoDepthBasicBlock
        else:
            raise NotImplementedError
        prob_0_L = (1, drop_prob)
        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(num_blocks) - 1)

        layers = []
        layers.append(ConvModule(in_channels, stem_channels, kernel_size=5, padding=2, norm_cfg=norm_cfg if stem_norm else None, act_cfg=act_cfg))
        self.loc2channels['l0'] = stem_channels

        inplanes = stem_channels
        for i in range(len(num_blocks)):
            planes = base_channels*plane_ratios[i]
            layers.append(self._make_layer(block, inplanes, planes, num_blocks[i], stride=strides[i]))
            inplanes = planes * block.expansion
            self.loc2channels[f'l{i+1}'] = inplanes
        self.layers = nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        img_downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            img_downsample = ConvModule(inplanes, planes*block.expansion, kernel_size=1, stride=stride, padding=0, norm_cfg=dict(type='BN'), act_cfg=None)

        m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
        img_layers = [block(self.prob_now, m, self.multFlag, inplanes, planes, stride, img_downsample)]
        self.prob_now = self.prob_now - self.prob_step

        for _ in range(1, blocks):
            m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
            img_layers.append(block(self.prob_now, m, self.multFlag, planes*block.expansion, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*img_layers)
