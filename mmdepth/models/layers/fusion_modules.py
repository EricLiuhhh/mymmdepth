import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
from .guides import CatConvGuide

@MODELS.register_module()
class CatConvFusion(CatConvGuide):
    def __init__(self, 
                 feat_planes, 
                 guide_planes, 
                 out_planes=None, 
                 normal_order=True, 
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 padding_mode='zeros', 
                 bias='auto', 
                 kernel_size=1,
                 proj_conv=None,
                 **kwargs):
        super().__init__(feat_planes, guide_planes, out_planes, normal_order, norm_cfg, act_cfg, padding_mode, bias, kernel_size, **kwargs)
        if proj_conv is not None:
            if isinstance(proj_conv, bool):
                self.conv_out = ConvModule(out_planes, out_planes, kernel_size, 1, kernel_size//2, norm_cfg=norm_cfg, act_cfg=act_cfg)
            elif isinstance(proj_conv, dict):
                self.conv_out = MODELS.build(proj_conv)
            else:
                raise NotImplementedError
        
    def forward(self, feat, weight):
        x = super().forward(feat, weight)
        x = self.conv_out(x)
        return x