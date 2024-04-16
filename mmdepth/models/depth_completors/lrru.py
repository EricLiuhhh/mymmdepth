from typing import List, Tuple
import copy
import torch
from mmengine.structures import PixelData
from torch import Tensor
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import OptConfigType, OptMultiConfig, ConfigType, OptSampleList, SampleList
from .base_completor import BaseCompletor

@MODELS.register_module()
class LRRU(BaseCompletor):
    def __init__(self, 
                 img_encoder: ConfigType,
                 depth_guide: ConfigType,
                 depth_encoder: ConfigType,
                 decode_guide: ConfigType,
                 decoder: ConfigType,
                 refinement: ConfigType,
                 loss_cfg: ConfigType,
                 data_preprocessor: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        assert img_encoder['type'] == depth_encoder['type']
        self.img_encoder = MODELS.build(img_encoder)
        self.depth_encoder = MODELS.build(depth_encoder)
        self.depth_guide_map = depth_guide['guide_map']
        self.guide_map[('img_encoder', list(zip(*self.depth_guide_map))[0])] = ('depth_encoder', list(zip(*self.depth_guide_map))[1])
        self.build_guide(depth_guide, self.img_encoder, self.depth_encoder)
        self.decoder = MODELS.build(decoder)
        self.decode_guide_map = decode_guide['guide_map']
        self.guide_map[('depth_encoder', list(zip(*self.decode_guide_map))[0])] = ('decoder', list(zip(*self.decode_guide_map))[1])
        self.build_guide(decode_guide, self.depth_encoder, self.decoder)
        self.refinement = MODELS.build(refinement)
        self.loss_func = MODELS.build(loss_cfg)

    def _forward(self, batch_inputs, batch_data_samples):
        img_encoder_inputs = dict(feats=batch_inputs['img'])
        img_encoder_outputs = self.img_encoder(img_encoder_inputs)

        depth_encoder_inputs = dict(feats = batch_inputs['sparse_depth'])
        for i, k in enumerate(self.depth_guide_map):
            depth_encoder_inputs[f'guide{i}'] = img_encoder_outputs[f'{k[0]}']
        depth_encoder_ouputs = self.depth_encoder(depth_encoder_inputs)

        decoder_inputs = dict(feats = depth_encoder_ouputs[f'l{len(self.depth_encoder.layers)-1}'] + img_encoder_outputs[f'l{len(self.img_encoder.layers)-1}'])
        for i, k in enumerate(self.decode_guide_map):
            decoder_inputs[f'guide{i}'] = depth_encoder_ouputs[f'{k[0]}']
        decoder_outputs = self.decoder(decoder_inputs)

        refinement_inputs = dict(
            coarse_depth = batch_inputs['prefill_depth'],
            sparse_depth = batch_inputs['filtered_depth'],
            guide0 = decoder_outputs['g0'],
            guide1 = decoder_outputs['g1'],
            guide2 = decoder_outputs['g2'],
            guide3 = decoder_outputs[f'l{len(self.decoder.layers)-1}']+depth_encoder_ouputs['l0']
        )
        preds = self.refinement(refinement_inputs)
        return preds
    
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        preds = self._forward(batch_inputs, batch_data_samples)
        losses = self.loss_func(preds, self._stack_batch_gt(batch_data_samples))
        return losses

    def predict(self, batch_inputs, batch_data_samples=None) -> SampleList:
        preds = self._forward(batch_inputs, batch_data_samples)
        y = torch.clamp(preds[-1], min=0)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].pred_depth = PixelData(**dict(data=y[i]))
        return batch_data_samples