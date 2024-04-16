from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from mmengine.structures import PixelData
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import OptConfigType, OptMultiConfig, ConfigType, OptSampleList, SampleList
from .base_completor import BaseCompletor

@MODELS.register_module()
class CompletionFormer(BaseCompletor):
    def __init__(self, 
                 encoder_decoder: ConfigType,
                 refinement: ConfigType,
                 loss_cfg: ConfigType,
                 data_preprocessor: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(encoder_decoder)
        self.refinement = MODELS.build(refinement)
        self.loss_func = MODELS.build(loss_cfg)

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        coarse = self.backbone(batch_inputs)
        refine = self.refinement(coarse)
        pred = torch.clamp(refine[0], min=0)
        return pred

    def loss(self, batch_inputs, batch_data_samples):
        pred = self._forward(batch_inputs, batch_data_samples)
        gt = self._stack_batch_gt(batch_data_samples)
        losses = self.loss_func(pred, gt)
        return losses
    
    def predict(self, batch_inputs, batch_data_samples=None) -> SampleList:
        pred = self._forward(batch_inputs, batch_data_samples)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].pred_depth = PixelData(**dict(data=pred[i]))
        return batch_data_samples