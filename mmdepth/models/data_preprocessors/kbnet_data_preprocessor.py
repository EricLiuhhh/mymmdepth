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
from mmdepth.datasets.transforms.utils import mindiff_outlier_removal
from .data_preprocessor import DepthDataPreprocessor

try:
    import skimage
except ImportError:
    skimage = None

def meshgrid(n_batch, n_height, n_width, device, homogeneous=True):
    '''
    Creates N x 2 x H x W meshgrid in x, y directions

    Arg(s):
        n_batch : int
            batch size
        n_height : int
            height of tensor
        n_width : int
            width of tensor
        device : torch.device
            device on which to create meshgrid
        homoegenous : bool
            if set, then add homogeneous coordinates (N x H x W x 3)
    Return:
        torch.Tensor[float32]: N x 2 x H x W meshgrid of x, y and 1 (if homogeneous)
    '''

    x = torch.linspace(start=0.0, end=n_width-1, steps=n_width, device=device)
    y = torch.linspace(start=0.0, end=n_height-1, steps=n_height, device=device)

    # Create H x W grids
    grid_y, grid_x = torch.meshgrid(y, x)

    if homogeneous:
        # Create 3 x H x W grid (x, y, 1)
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

    grid_xy = torch.unsqueeze(grid_xy, dim=0) \
        .repeat(n_batch, 1, 1, 1)

    return grid_xy

def camera_coordinates(batch, height, width, k):
    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = meshgrid(
        n_batch=batch,
        n_height=height,
        n_width=width,
        device=k.device,
        homogeneous=True)
    xy_h = xy_h.view(batch, 3, -1)

    # K^-1 [x, y, 1] z and reshape back to N x 3 x H x W
    coordinates = torch.matmul(torch.inverse(k), xy_h)
    coordinates = coordinates.view(batch, 3, height, width)

    return coordinates

def scale_intrinsics(batch, height0, width0, height1, width1, k):
    device = k.device

    width0 = torch.tensor(width0, dtype=torch.float32, device=device)
    height0 = torch.tensor(height0, dtype=torch.float32, device=device)
    width1 = torch.tensor(width1, dtype=torch.float32, device=device)
    height1 = torch.tensor(height1, dtype=torch.float32, device=device)

    # Get scale in x, y components
    scale_x = width1 / width0
    scale_y = height1 / height0

    # Prepare 3 x 3 matrix to do element-wise scaling
    scale = torch.tensor([[scale_x,     1.0, scale_x],
                            [1.0,     scale_y, scale_y],
                            [1.0,         1.0,      1.0]], dtype=torch.float32, device=device)

    scale = scale.view(1, 3, 3).repeat(batch, 1, 1)

    return k * scale
    
@MODELS.register_module()
class KBNetDataPreprocessor(DepthDataPreprocessor):
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None,
                 strides: Optional[List[int]] = [1, 2, 2, 2],
                 ):
        super().__init__(mean, std, bgr_to_rgb, rgb_to_bgr, non_blocking, batch_augments)
        self.strides = strides


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
        img = inputs['img']
        K = inputs['K'] # N 1 3 3
        h0, w0 = img.shape[-2:]
        h, w = h0, w0
        b = img.shape[0]
        coords = []
        for stride in self.strides:
            if stride == 1:
                coords.append(camera_coordinates(b, h, w, K.squeeze(1)))
            else:
                h, w = h // stride, w // stride
                #intrinsics = scale_intrinsics(b, h0, w0, h, w, K.squeeze(1))
                # original bug
                intrinsics = scale_intrinsics(b, h0, w0, h0//2, w0//2, K.squeeze(1))
                coords.append(camera_coordinates(b, h, w, intrinsics))
        data['inputs']['pos_embds'] = coords

        _, data['inputs']['valid_map'] = mindiff_outlier_removal(data['inputs']['sparse_depth'])
        return data
