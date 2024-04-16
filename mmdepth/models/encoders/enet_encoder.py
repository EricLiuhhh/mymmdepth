from typing import List, Union
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule, build_norm_layer
from mmdepth.registry import MODELS
from .base_guided_net import BaseGuidedNet
    
class BasicBlockGeo(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_cfg=dict(type='BN'), geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes + geoplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes+geoplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes+geoplanes, planes, 1, stride, 0, bias=False),
                build_norm_layer(norm_cfg, planes)[1],
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicLayerGeo(nn.Sequential):
    def __init__(self, *blocks):
        if len(blocks) > 0: assert isinstance(blocks[0], BasicBlockGeo), 'Block type should be BasicBlockGeo.'
        super().__init__(*blocks)

    def forward(self, x, geo1, geo2):
        for block in self:
            if block.downsample:
                x = block(x, geo1, geo2)
            else:
                x = block(x, geo2, geo2)
        return x

@MODELS.register_module()
class ENetEncoder(BaseGuidedNet):
    def __init__(self,
                 geoplanes,
                 in_channels=4,
                 out_channels=2,
                 base_channels=32,
                 plane_ratios=(2, 2, 2, 2, 2),
                 num_blocks=(2, 2, 2, 2, 2),
                 custom_inplanes=None,
                 conv_out_cfg=None,
                 self_guide_map=None,
                 hook_before_self_guide=False,
                 deconv_cfg=dict(type='ConvT', output_padding=1),
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 hook_positions=None,
                 init_cfg=None):
        assert len(plane_ratios) == len(num_blocks)
        num_stage = len(num_blocks)
        hook_positions = dict(l=tuple(range(num_stage, num_stage*2+2))) if hook_positions is None else hook_positions
        if self_guide_map is None:
            self_guide_map = dict([(f'l{i}', f'l{num_stage*2-i}') for i in range(num_stage)])
        super().__init__(hook_positions=hook_positions, self_guide_map=self_guide_map, hook_before_self_guide=hook_before_self_guide, init_cfg=init_cfg)

        self.geoplanes = geoplanes
        self.num_stage = num_stage
        self.num_blocks = num_blocks
        
        layers = []
        # stem
        layers.append(ConvModule(in_channels=in_channels, out_channels=base_channels, kernel_size=5, stride=1, padding=2, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.loc2channels[f'l{0}'] = base_channels

        # blocks
        inplanes = base_channels
        for i in range(num_stage):
            if custom_inplanes is not None:
                inplanes = custom_inplanes[i]
            planes = inplanes*plane_ratios[i]
            single_layer = []
            for k in range(num_blocks[i]):
                single_layer.append(BasicBlockGeo(inplanes, planes, stride=2 if k==0 else 1, geoplanes=self.geoplanes))
                inplanes = planes*BasicBlockGeo.expansion
            single_layer = BasicLayerGeo(*single_layer)
            self.loc2channels[f'l{i+1}'] = inplanes
            layers.append(single_layer)

        # decode
        for i in range(num_stage):
            planes = inplanes // num_blocks[num_stage-i-1]
            layers.append(ConvModule(in_channels=inplanes, out_channels=planes, kernel_size=5, stride=2, padding=2, conv_cfg=deconv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            inplanes = planes
            self.loc2channels[f'l{i+1+num_stage}'] = inplanes

        # conv out
        layers.append(ConvModule(in_channels=inplanes, out_channels=out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, conv_cfg=conv_out_cfg))
        self.loc2channels[f'l{1+num_stage*2}'] = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        _inputs = inputs
        for i in range(self.num_stage):
            _inputs[f'ext_feats_l{i+1}'] = (inputs[f'geo_embds{i}'], inputs[f'geo_embds{i+1}'])

        return super().forward(_inputs)