import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdepth.models.encoders import BaseGuidedNet
from mmdepth.registry import MODELS

class VGGNetBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20),
                 norm_cfg=None,
                 bias='auto'):
        super(VGGNetBlock, self).__init__()
        conv2d = ConvModule

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                bias=bias)
            layers.append(conv)
            in_channels = out_channels

        conv = conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=bias)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through a VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)
    
@MODELS.register_module()
class KBNetEncoder(BaseGuidedNet):
    def __init__(self,
                 in_channels=3,
                 planes=[48, 96, 192, 384, 384],
                 num_convs_per_block=[1, 1, 1, 1, 1],
                 strides=[2, 2, 2, 2, 2],
                 backproj_layers=[0, 1, 2, 3],
                 ext_feats_channels=0,
                 coord_guide=False,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20)):
        super(KBNetEncoder, self).__init__()
        self.backproj_layers = backproj_layers
        self.n_filters = planes
        layers = []
        self.num_stages = len(planes)

        inplanes = in_channels
        layer_cnt = 0

        for i in range(self.num_stages):
            if i in backproj_layers and i == 0:
                layers.append(ConvModule(
                    in_channels=inplanes,
                    out_channels=planes[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_cfg=act_cfg,
                    norm_cfg=None,
                    bias=False))
                inplanes = planes[0]
                self.loc2channels[f'l{layer_cnt}'] = inplanes
                layer_cnt += 1
            layers.append(VGGNetBlock(
            in_channels=inplanes+ext_feats_channels if i in backproj_layers else inplanes,
            out_channels=planes[i],
            n_convolution=num_convs_per_block[i],
            stride=strides[i],
            act_cfg=act_cfg,
            bias=False))
            inplanes = planes[i]
            self.loc2channels[f'l{layer_cnt}'] = inplanes
            layer_cnt += 1
        self.layers = nn.Sequential(*layers)
        self.hook_positions = dict(l=tuple(range(len(layers))))
        if coord_guide:
            guides = [dict(type='CatGuide', feat_planes=self.loc2channels[f'l{i}'], guide_planes=3) for i in backproj_layers]
            guide_locations = [f'l{i}' for i in backproj_layers]
            self.add_guides(guides, guide_locations)