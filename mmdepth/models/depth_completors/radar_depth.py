from typing import Dict
from torch import Tensor
import torch.nn as nn
from mmengine.structures import PixelData
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig, SampleList
from .dual_branch_completor import DualBranchCompletor

@MODELS.register_module()
class RadarDepth(DualBranchCompletor):
    def __init__(self, 
                 img_encoder: ConfigType, 
                 depth_encoder: ConfigType, 
                 fusion_module: ConfigType, 
                 decoder: ConfigType,
                 loss_cfg: ConfigType,
                 output_size,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(img_encoder, depth_encoder, fusion_module, decoder, data_preprocessor, init_cfg)

        self.conv_out = nn.Conv2d(self.decoder.get_planes(-1), 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=output_size[::-1], mode='bilinear', align_corners=True)
        self.loss_func = MODELS.build(loss_cfg)

    def loss(self, inputs: Tensor, batch_data_samples):
        out = self._forward(inputs)
        out = self.conv_out(self.get_last_feat(out))
        pred = self.bilinear(out)
        gt = self._stack_batch_gt(batch_data_samples)
        losses = self.loss_func(pred, gt)
        return losses

    def predict(self, inputs: Tensor, batch_data_samples = None) -> SampleList:
        out = self._forward(inputs)
        out = self.conv_out(self.get_last_feat(out))
        pred = self.bilinear(out)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].pred_depth = PixelData(**dict(data=pred[i].detach()))
        return batch_data_samples