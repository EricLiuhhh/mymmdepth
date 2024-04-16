from typing import List, Union
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out
    
class BasicDepthEncoder(nn.Module):

    def __init__(self, kernel_size, block=BasicBlock, base_channels=16, norm_layer=nn.BatchNorm2d):
        super(BasicDepthEncoder, self).__init__()
        self._norm_layer = norm_layer
        self.kernel_size = kernel_size
        self.num = kernel_size*kernel_size - 1
        self.idx_ref = self.num // 2
        bc = base_channels

        self.convd1 = ConvModule(1, bc, kernel_size=3, padding=1, norm_cfg=None)
        self.convd2 = ConvModule(bc, bc, kernel_size=3, padding=1, norm_cfg=None)

        self.convf1 = ConvModule(bc, bc, kernel_size=3, padding=1, norm_cfg=None)
        self.convf2 = ConvModule(bc, bc, kernel_size=3, padding=1, norm_cfg=None)

        self.conv = ConvModule(bc * 2, bc * 2, kernel_size=3, padding=1, norm_cfg=None)
        self.ref = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)
        self.conv_weight = nn.Conv2d(bc * 2, self.kernel_size**2, kernel_size=1, stride=1, padding=0)
        self.conv_offset = nn.Conv2d(bc * 2, 2*(self.kernel_size**2 - 1), kernel_size=1, stride=1, padding=0)

    def forward(self, depth, context):
        B, _, H, W = depth.shape

        d1 = self.convd1(depth)
        d2 = self.convd2(d1)

        f1 = self.convf1(context)
        f2 = self.convf2(f1)

        input_feature = torch.cat((d2, f2), dim=1)
        input_feature = self.conv(input_feature)
        feature = self.ref(input_feature)
        weight = torch.sigmoid(self.conv_weight(feature))
        offset = self.conv_offset(feature)

        # Add zero reference offset
        offset = offset.view(B, self.num, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num, dim=1))
        list_offset.insert(self.idx_ref,
                           torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        return weight, offset
    
class LRRUCSPN(nn.Module):

    def __init__(self, 
                 kernel_size=3,
                 residual=True,
                 ):
        super().__init__()

        self.dkn_residual = residual

        self.w = nn.Parameter(torch.ones((1, 1, kernel_size, kernel_size)))
        self.b = nn.Parameter(torch.zeros(1))
        self.stride = 1
        self.padding = int((kernel_size - 1) / 2)
        self.dilation = 1
        self.deformable_groups = 1
        self.im2col_step = 64

    def forward(self, depth, weight, offset):

        if self.dkn_residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        output = deform_conv2d(
            depth, offset, weight=self.w, bias=self.b, stride=self.stride,
            padding=self.padding, dilation=self.dilation, mask=weight)

        if self.dkn_residual:
            output = output + depth

        return output
    
@MODELS.register_module()
class LRRUTDU(nn.Module):
    def __init__(self,
                 base_channels=32,
                 plane_ratios=(8, 4, 2, 1),
                 strides=(2, 2, 2),
                 num_stage=3,
                 ) -> None:
        super().__init__()
        assert num_stage == len(strides) == len(plane_ratios)-1
        self.up_blocks = []
        deconv_cfg = dict(type='ConvT', output_padding=1)
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')
        for i in range(num_stage):
            in_channels = base_channels*plane_ratios[i]
            out_channels = base_channels*plane_ratios[i+1]
            self.up_blocks.append(ConvModule(in_channels, out_channels, 3, 2, 1, conv_cfg=deconv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.up_blocks = nn.Sequential(*self.up_blocks)
        self.weight_offset = BasicDepthEncoder(kernel_size=3, block=BasicBlock, base_channels=base_channels, norm_layer=nn.BatchNorm2d)

    def forward(self, inputs):
        x, guide = inputs['feats'], inputs['guide']
        guide_up = self.up_blocks(guide)
        weight, offset = self.weight_offset(x, guide_up)
        return weight, offset
    
@MODELS.register_module()
class LRRURefinement(BaseModule):
    def __init__(self, 
                 base_channels=32,
                 preserve_depth=True,
                 depth_norm=False,
                 num_stage=(3, 2, 1, 0),
                 plane_ratios=((8, 4, 2, 1), (8, 4, 1), (4, 1), (1,)),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.preserve_input = preserve_depth
        self.depth_norm = depth_norm
        self.layers = []
        self.num_stage = len(num_stage)
        for i in range(len(num_stage)):
            self.layers.append(LRRUTDU(base_channels, plane_ratios=plane_ratios[i], strides=[2]*num_stage[i], num_stage=num_stage[i]))
        self.layers = nn.Sequential(*self.layers)
        self.cspn = LRRUCSPN()

    def forward(self, inputs):
        sparse_depth = inputs['sparse_depth']
        coarse_depth = inputs['coarse_depth']
        results = []
        refined_depth = coarse_depth
        for i in range(self.num_stage):
            if self.preserve_input:
                mask = torch.sum(sparse_depth > 0.0, dim=1, keepdim=True)
                mask = (mask > 0.0).type_as(sparse_depth)
                refined_depth = (1.0 - mask) * refined_depth + mask * sparse_depth
            refined_depth = refined_depth.detach()
            tdu_inputs = dict(feats=refined_depth, guide=inputs[f'guide{i}'])
            weight, offset = self.layers[i](tdu_inputs)
            refined_depth = self.cspn(refined_depth, weight, offset)
            results.append(refined_depth)

        return results