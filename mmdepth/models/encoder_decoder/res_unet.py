from typing import List, Union
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, ConvModule
from mmcv.cnn.resnet import ResNet
from mmengine.model import BaseModule, ModuleList, Sequential

from mmdepth.registry import MODELS


@MODELS.register_module()
class ResUNet(BaseModule):
    def __init__(self,
                 prop_kernel_size=3,
                 confidence_branch=True,
                 res_layer_type='resnet34',
                 conv_cfg=None,
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

        if res_layer_type == 'resnet18':
            net = ResNet(depth=18)
        elif res_layer_type == 'resnet34':
            net = ResNet(depth=34)
        else:
            raise NotImplementedError

        for i, layer_name in enumerate(net.res_layers):
            res_layer = getattr(net, layer_name)
            self.__setattr__(f'conv{i+2}', res_layer)

        del net

        # 1/16
        self.conv6 = ConvModule(512, 512, kernel_size=3, stride=2, padding=1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)

        # Shared Decoder
        # 1/8
        self.dec5 = ConvModule(512, 256, kernel_size=3, stride=2,
                                  padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)
        # 1/4
        self.dec4 = ConvModule(256+512, 128, kernel_size=3, stride=2,
                                  padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)
        # 1/2
        self.dec3 = ConvModule(128+256, 64, kernel_size=3, stride=2,
                                  padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)

        # 1/1
        self.dec2 = ConvModule(64+128, 64, kernel_size=3, stride=2,
                                  padding=1, conv_cfg=deconv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg)

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

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f
    
    def forward(self, inputs):
        rgb = inputs['img']
        dep = inputs['sparse_depth']

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)
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
            'feat_init': pred_init,
            'affinity': guide,
            'confidence': confidence,
            'feat_fix': dep,
            'img': rgb
        }
        return results