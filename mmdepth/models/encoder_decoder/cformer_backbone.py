from typing import List, Union
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, ConvModule
from mmcv.cnn.resnet import ResNet
from mmengine.model import BaseModule, ModuleList, Sequential

from mmdepth.registry import MODELS


@MODELS.register_module()
class CFormerBackbone(BaseModule):
    def __init__(self,
                 prop_kernel_size=3,
                 confidence_branch=True,
                 conv_cfg=None,
                 pvt_pretrain=None,
                 deconv_cfg=dict(type='ConvT', output_padding=1),
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_neighbors = prop_kernel_size*prop_kernel_size-1
        self.use_conf = confidence_branch

        # Encoder
        self.conv1_rgb = ConvModule(3, 48, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=None)
        self.conv1_dep = ConvModule(1, 16, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=None)
        self.conv1 = ConvModule(64, 64, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=None)

        self.former = MODELS.build(dict(type='PVT', in_chans=64, patch_size=2, pretrained=pvt_pretrain))
        
        channels = [64, 128, 64, 128, 320, 512]
        basic_block = MODELS.get('CBAMBasicBlock')
        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            ConvModule(channels[5], 256, kernel_size=3, stride=2, padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg),
            basic_block(256, 256, stride=1, downsample=None, ratio=16),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            ConvModule(256+channels[4], 128, kernel_size=3, stride=2, padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg),
            basic_block(128, 128, stride=1, downsample=None, ratio=8),
        )
        # 1/4
        self.dec4 = nn.Sequential(
            ConvModule(128+channels[3], 64, kernel_size=3, stride=2, padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg),
            basic_block(64, 64, stride=1, downsample=None, ratio=4),
        )
        # 1/2
        self.dec3 = nn.Sequential(
            ConvModule(64+channels[2], 64, kernel_size=3, stride=2, padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg),
            basic_block(64, 64, stride=1, downsample=None, ratio=4),
        )
        # 1/1
        self.dec2 = nn.Sequential(
            ConvModule(64+channels[1], 64, kernel_size=3, stride=2, padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg),
            basic_block(64, 64, stride=1, downsample=None, ratio=4),
        )

        # Init Depth Branch
        # 1/1
        self.id_dec1 = ConvModule(64+64, 64, kernel_size=3, stride=1,
                                    padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.id_dec0 = ConvModule(64+64, 1, kernel_size=3, stride=1,
                                    padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=None)

        # Guidance Branch
        # 1/1
        self.gd_dec1 = ConvModule(64+64, 64, kernel_size=3, stride=1,
                                    padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.gd_dec0 = ConvModule(64+64, self.num_neighbors, kernel_size=3, stride=1,
                                    padding=1, conv_cfg=conv_cfg, act_cfg=None, norm_cfg=None)

        if confidence_branch:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = ConvModule(64+64, 32, kernel_size=3, stride=1,
                                        padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)
            self.cf_dec0 = ConvModule(32+64, 1, kernel_size=3, stride=1,
                                        padding=1, conv_cfg=conv_cfg, act_cfg=dict(type='Sigmoid'), norm_cfg=None)

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f
    
    def forward(self, inputs):
        rgb = inputs['img']
        dep = inputs['sparse_depth']

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        fe1 = self.conv1(fe1)

        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.use_conf:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None
        
        results = {
            'feat_init': pred_init + dep,
            'affinity': guide,
            'confidence': confidence,
            'feat_fix': dep,
            'img': rgb
        }
        return results