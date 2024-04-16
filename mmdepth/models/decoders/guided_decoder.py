from typing import List, Union
import math
import torch
import torch.nn as nn
from torch.nn import init
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
from mmdepth.models.encoders import BaseGuidedNet


@MODELS.register_module()
class GuidedDecoder(BaseGuidedNet):
    def __init__(self, 
                 base_channels=32,
                 num_stage=5,
                 plane_ratios=(8, 8, 8, 4, 2, 1),
                 strides=(2, 2, 2, 2, 1),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 padding_mode='zeros',
                 guides=None,
                 guide_locations=None,
                 hook_positions={'l':(0, 1, 2, 3, 4)},
                 upsample_type='convt',
                 input_upsample_type=None,
                 init_cfg=None):
        super().__init__(guides, guide_locations, hook_positions, init_cfg)
        if isinstance(plane_ratios[0], int): assert num_stage == len(strides) == (len(plane_ratios)-1)
        layers = []
        scale = 1
        for i in range(num_stage):
            scale *= strides[i]
            if isinstance(plane_ratios[0], int):
                in_channels = base_channels*plane_ratios[i]
                out_channels = base_channels*plane_ratios[i+1]
            else:
                in_channels = base_channels*plane_ratios[i][0]
                out_channels = base_channels*plane_ratios[i][1]
            if strides[i] == 1:
                layers.append(ConvModule(in_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode))
                self.loc2channels[f'l{i}'] = out_channels
                self.loc2scales[f'l{i}'] = scale
                continue
            if i == 0 and input_upsample_type is not None:
                _upsample_type = input_upsample_type
            else:
                _upsample_type = upsample_type
            if _upsample_type == 'convt':
                deconv_cfg = dict(type='ConvT', output_padding=1)
                layers.append(ConvModule(in_channels, out_channels, 3, strides[i], 1, conv_cfg=deconv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode))
            elif _upsample_type == 'S2DUpBlock':
                block = MODELS.build(dict(type=_upsample_type, in_channels=in_channels, out_channels=out_channels))
                layers.append(block)
            elif _upsample_type == 'nearest':
                assert in_channels == out_channels
                layers.append(nn.UpsamplingNearest2d(scale_factor=strides[i]))
            elif _upsample_type == 'bilinear':
                assert in_channels == out_channels
                layers.append(nn.UpsamplingBilinear2d(scale_factor=strides[i]))
            elif _upsample_type == 'nearest_conv':
                layers.append(nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=strides[i]),
                    ConvModule(in_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode)
                ))
            elif _upsample_type == 'bilinear_conv':
                layers.append(nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=strides[i]),
                    ConvModule(in_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode)
                ))
            elif _upsample_type == 'conv_nearest':
                layers.append(nn.Sequential(
                    ConvModule(in_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode),
                    nn.UpsamplingNearest2d(scale_factor=strides[i])
                ))
            elif _upsample_type == 'conv_bilinear':
                layers.append(nn.Sequential(
                    ConvModule(in_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode),
                    nn.UpsamplingBilinear2d(scale_factor=strides[i])
                ))
            else:
                raise NotImplementedError
            self.loc2channels[f'l{i}'] = out_channels
            self.loc2scales[f'l{i}'] = scale
        self.layers = nn.Sequential(*layers)
        self.padding_mode = padding_mode
    
    def build_conv_outs(self, conv_out_locations, ks=3, out_channels=1):
        self.conv_outs = nn.ModuleDict()
        self.conv_out_locations = conv_out_locations
        if isinstance(out_channels, int):
            out_channels = [out_channels] * len(self.conv_out_locations)
        for i, loc in enumerate(conv_out_locations):
            self.conv_outs[loc] = ConvModule(self.loc2channels[loc], out_channels[i], ks, 1, ks//2, norm_cfg=None, act_cfg=None, padding_mode=self.padding_mode)

    def forward(self, inputs, stop_at=None):
        results = super().forward(inputs, stop_at)
        if hasattr(self, 'conv_outs'):
            for k, v in results.items():
                if k in self.conv_out_locations:
                    results[k] = self.conv_outs[k](v)
        return results