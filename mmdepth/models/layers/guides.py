import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmdepth.registry import MODELS
from mmdepth.models.layers import Conv2dGuided

@MODELS.register_module()
class CatConvGuide(nn.Module):
    def __init__(self, 
                 feat_planes,
                 guide_planes,
                 out_planes=None,
                 normal_order=True,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 padding_mode='zeros',
                 bias='auto',
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        if out_planes is None:
            out_planes = feat_planes
        self.out_channels = out_planes
        self.conv = ConvModule(feat_planes+guide_planes, out_planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, norm_cfg=norm_cfg, act_cfg=act_cfg, bias=bias, padding_mode=padding_mode)
        self.normal_order = normal_order

    def forward(self, feat, weight):
        if weight is None:
            x = feat
        else:
            if self.normal_order:
                x = torch.cat((feat, weight), dim=1)
            else:
                x = torch.cat((weight, feat), dim=1)
        x = self.conv(x)
        return x
    
@MODELS.register_module()
class AddGuide(nn.Module):
    def __init__(self, feat_planes, guide_planes):
        super().__init__()
        assert feat_planes == guide_planes
        self.out_channels = feat_planes

    def forward(self, feat, guide):
        return feat + guide
    
@MODELS.register_module()
class CatGuide(nn.Module):
    def __init__(self, feat_planes, guide_planes, normal_order=True):
        super().__init__()
        self.out_channels = feat_planes + guide_planes
        self.normal_order = normal_order

    def forward(self, feat, guide):
        if guide is None:
            return feat
        # NCHW
        if self.normal_order:
            return torch.cat((feat, guide), dim=1)
        else:
            return torch.cat((guide, feat), dim=1)

        
@MODELS.register_module()
class KernelLearningGuide(nn.Module):
    

    def __init__(self, 
                 feat_planes, 
                 guide_planes, 
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'), 
                 weight_ks=1
                 ):
        super().__init__()
        input_planes = feat_planes
        weight_planes = guide_planes
        self.local = nn.Sequential(
            Conv2dGuided(),
            build_norm_layer(norm_cfg, input_planes)[1],
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = ConvModule(input_planes + weight_planes, input_planes, 3, 1, 1, norm_cfg=None, act_cfg=act_cfg)
        self.conv12 = nn.Conv2d(input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2)
        self.conv21 = ConvModule(input_planes + weight_planes, input_planes, 3, 1, 1, norm_cfg=None, act_cfg=act_cfg)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            build_norm_layer(norm_cfg, num_features=input_planes)[1],
            nn.ReLU(inplace=True),
        )
        self.conv3 = ConvModule(input_planes, input_planes, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.out_channels = input_planes

    def forward(self, input, weight):
        B, Ci, H, W = input.shape
        weight = torch.cat([input, weight], 1)
        weight11 = self.conv11(weight)
        weight12 = self.conv12(weight11)
        weight21 = self.conv21(weight)
        weight21 = self.pool(weight21)
        weight22 = self.conv22(weight21).view(B, -1, Ci)
        out = self.local([input, weight12]).view(B, Ci, -1)
        out = torch.bmm(weight22, out).view(B, Ci, H, W)
        out = self.br(out)
        out = self.conv3(out)
        return out