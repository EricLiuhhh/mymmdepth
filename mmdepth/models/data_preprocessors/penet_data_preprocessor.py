# Copyright (c) OpenMMLab. All rights reserved.
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor
from mmdepth.registry import MODELS
from .data_preprocessor import DepthDataPreprocessor

try:
    import skimage
except ImportError:
    skimage = None

class GeometryFeature(nn.Module):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z*(0.5*h*(vnorm+1)-ch)/fh
        y = z*(0.5*w*(unorm+1)-cw)/fw
        return torch.cat((x, y, z),1)

class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d
        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number
        return d_result, mask_result
    
@MODELS.register_module()
class PENetDataPreprocessor(DepthDataPreprocessor):
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None,
                 img_shape = (352, 1216),
                 num_stage = 6,
                 geo_encoding_type = 'xyz'
                 ):
        super().__init__(mean=mean, 
                         std=std, 
                         bgr_to_rgb=bgr_to_rgb, 
                         rgb_to_bgr=rgb_to_bgr, 
                         non_blocking=non_blocking, 
                         batch_augments=batch_augments)
        self.num_stage = num_stage
        self.img_shape = img_shape
        self.geo_encoding_type = geo_encoding_type
        self.geofeature = None
        self.geoplanes = 3
        if geo_encoding_type == 'xyz':
            self.geofeature = GeometryFeature()
        elif geo_encoding_type == "std":
            self.geoplanes = 0
        elif geo_encoding_type == "uv":
            self.geoplanes = 2
        elif geo_encoding_type == "z":
            self.geoplanes = 1
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.collate_data(data)

        inputs = data['inputs']
        d = inputs['sparse_depth']

        position = inputs['pos_embd']
        K = inputs['K'][0]
        unorm = position[:, 0:1, :, :]
        vnorm = position[:, 1:2, :, :]

        fx = K[:, 0, 0].view((1, 1, 1, 1))
        cx = K[:, 0, 2].view((1, 1, 1, 1))
        fy = K[:, 1, 1].view((1, 1, 1, 1))
        cy = K[:, 1, 2].view((1, 1, 1, 1))

        vnorms = [vnorm]
        unorms = [unorm]
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        downsampled_depth = [d]
        downsampled_mask = [valid_mask]
        for _ in range(self.num_stage-1):
            vnorms.append(self.pooling(vnorms[-1]))
            unorms.append(self.pooling(unorms[-1]))
            temp = self.sparsepooling(downsampled_depth[-1], downsampled_mask[-1])
            downsampled_depth.append(temp[0])
            downsampled_mask.append(temp[1])

        geo_embds = []
        h, w = self.img_shape
        for i in range(self.num_stage):
            if self.geo_encoding_type == 'xyz':
                geo_embds.append(self.geofeature(downsampled_depth[i], vnorms[i], unorms[i], h/(2**i), w/(2**i), cy, cx, fy, fx))
            elif self.geo_encoding_type == 'uv':
                geo_embds.append(torch.cat(vnorms[i], unorms[i]))
            elif self.geo_encoding_type == 'z':
                geo_embds.append(downsampled_depth[i])
            data['inputs'][f'geo_embds{i}'] = geo_embds[-1]

        return data
