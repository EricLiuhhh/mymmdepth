from typing import List, Optional
import torch
from mmengine.structures import BaseDataElement, PixelData

class DepthDataSample(BaseDataElement):
    @property
    def gt_depth(self):
        return self._gt_depth

    @gt_depth.setter
    def gt_depth(self, value):
        self.set_field(value, '_gt_depth', dtype=PixelData)

    @gt_depth.deleter
    def gt_depth(self):
        del self._gt_depth

    @property
    def pred_depth(self):
        return self._pred_depth

    @pred_depth.setter
    def pred_depth(self, value):
        self.set_field(value, '_pred_depth', dtype=PixelData)

    @pred_depth.deleter
    def pred_depth(self):
        del self._pred_depth
    