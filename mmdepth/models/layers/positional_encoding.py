import math

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import MultiConfig, OptMultiConfig


@MODELS.register_module()
class PositionalEncoding(BaseModule):
    """Position encoding

    Args:
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 normalize: bool = True,
                 scale: float = 1,
                 offset: float = 0.,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.normalize = normalize
        self.scale = scale
        self.offset = offset

    def forward(self, mask: Tensor) -> Tensor:
        """Forward function for `PositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, 2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)-1
        x_embed = not_mask.cumsum(2, dtype=torch.float32)-1
        if self.normalize:
            # (0, 1) -> (0, scale) -> (offset, scale+offset)
            y_embed = (y_embed / y_embed[:, -1:, :]) * self.scale + self.offset
            x_embed = (x_embed / x_embed[:, :, -1:]) * self.scale + self.offset
        pos_x = x_embed[:, :, :, None]
        pos_y = y_embed[:, :, :, None]
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        pos = torch.cat((pos_x, pos_y), dim=3)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        return repr_str