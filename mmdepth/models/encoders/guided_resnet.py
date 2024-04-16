#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:50 PM
import torch
import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo as model_zoo
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
from .base_guided_net import BaseGuidedNet
from .resnet import BasicBlock

@MODELS.register_module()
class GuidedResNet(BaseGuidedNet):
    """
    Not activate at the ref
    Init change to trunctated norm
    """

    def __init__(self, 
                 in_channels=3,
                 blocktype='BasicBlock',
                 guides=None,
                 guide_locations=None,
                 hook_positions={'l':(1, 2, 3, 4, 5)},
                 base_channels=32,
                 stem_channels=None,
                 stem_norm=True,
                 num_blocks=(2, 2, 2, 2, 2),
                 plane_ratios=(2, 4, 8, 8, 8),
                 strides=(1, 2, 2, 2, 2),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 std_stem=False,
                 init_cfg=None):
        super().__init__(guides=guides, guide_locations=guide_locations, hook_positions=hook_positions, init_cfg=init_cfg)
        assert len(num_blocks) == len(plane_ratios) == len(strides)
        if stem_channels is None:
            stem_channels = base_channels
        if blocktype == 'BasicBlock':
            block = BasicBlock
        else:
            raise NotImplementedError
        self.norm_cfg = norm_cfg

        layers = []
        if not std_stem:
            layers.append(ConvModule(in_channels, stem_channels, kernel_size=5, padding=2, norm_cfg=norm_cfg if stem_norm else None, act_cfg=act_cfg))
        else:
            stem = nn.Sequential()
            stem.add_module('conv1', ConvModule(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, norm_cfg=norm_cfg, act_cfg=act_cfg))
            stem.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(stem)
        self.loc2channels['l0'] = stem_channels

        inplanes = stem_channels
        for i in range(len(num_blocks)):
            planes = base_channels*plane_ratios[i]
            layers.append(self._make_layer(block, inplanes, planes, num_blocks[i], stride=strides[i]))
            inplanes = planes * block.expansion
            self.loc2channels[f'l{i+1}'] = inplanes
        self.layers = nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(inplanes, planes * block.expansion, 1, stride, 0, act_cfg=None, norm_cfg=self.norm_cfg)

        layers = []
        layers.append(block(inplanes, planes, stride=stride, downsample=downsample, norm_cfg=self.norm_cfg))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_cfg=self.norm_cfg))

        return nn.Sequential(*layers)

@MODELS.register_module()
class StdResNet(BaseGuidedNet):
    def __init__(self, 
                 depth=18, 
                 pretrained=False, 
                 hook_positions=dict(l=(0, 1, 2, 3, 4, 5)),
                 num_input_images=1):
        super().__init__(hook_positions=hook_positions)

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if depth not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(depth))

        self.encoder = resnets[depth](pretrained)
        if num_input_images > 1:
            self.encoder.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(depth)])
            if num_input_images > 1:
                loaded['conv1.weight'] = torch.cat(
                [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
            self.encoder.load_state_dict(loaded)
        
        conv_in = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
        )
        self.layers = nn.Sequential(
            conv_in,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        )
        base_channels = 64
        scales = (2, 4, 4, 8, 16, 32)
        for i in range(len(self.layers)):
            self.loc2channels[f'l{i}'] = base_channels * 2**(i-2) if i > 2 else base_channels
            self.loc2scales[f'l{i}'] = 1 / scales[i]
        del self.encoder

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
