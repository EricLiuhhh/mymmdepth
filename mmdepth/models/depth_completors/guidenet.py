from typing import List, Tuple
import copy
import torch
import torch.nn as nn
from mmengine.structures import PixelData
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import OptConfigType, OptMultiConfig, ConfigType, OptSampleList, SampleList
from .base_completor import BaseCompletor
from ..encoders.resnet import BasicBlock

@MODELS.register_module()
class GuideNet(BaseCompletor):
    def __init__(self, 
                 img_encoder: ConfigType,
                 img_decoder: ConfigType,
                 img_decode_guide: ConfigType,
                 depth_guide: ConfigType,
                 depth_encoder: ConfigType,
                 decode_guide: ConfigType,
                 decoder: ConfigType,
                 data_preprocessor: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        assert img_encoder['type'] == depth_encoder['type']
        self.img_encoder = MODELS.build(img_encoder)
        self.img_decoder = MODELS.build(img_decoder)
        self.img_decode_guide_map = img_decode_guide['guide_map']
        self.guide_map[('img_encoder', list(zip(*self.img_decode_guide_map))[0])] = ('img_decoder', list(zip(*self.img_decode_guide_map))[1])
        self.build_guide(img_decode_guide, self.img_encoder, self.img_decoder)

        self.depth_encoder = MODELS.build(depth_encoder)
        self.depth_guide_map = depth_guide['guide_map']
        self.guide_map[('img_decoder', list(zip(*self.depth_guide_map))[0])] = ('depth_encoder', list(zip(*self.depth_guide_map))[1])
        self.build_guide(depth_guide, self.img_decoder, self.depth_encoder)

        self.decoder = MODELS.build(decoder)
        self.decode_guide_map = decode_guide['guide_map']
        self.guide_map[('depth_encoder', list(zip(*self.decode_guide_map))[0])] = ('decoder', list(zip(*self.decode_guide_map))[1])
        self.build_guide(decode_guide, self.depth_encoder, self.decoder)

        ch = self.decoder.get_planes(-1)
        self.conv_out = nn.Sequential(
            BasicBlock(ch, ch, act_out=False),
            nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1)
        )

    def loss(self, batch_inputs, batch_data_samples):
        pass
    
    def _forward(self, batch_inputs, batch_data_samples=None):
        pass

    
    def predict(self, batch_inputs, batch_data_samples=None) -> SampleList:
        img_encoder_inputs = dict(feats=batch_inputs['img'])
        img_encoder_outputs = self.img_encoder(img_encoder_inputs)

        img_decoder_inputs = dict(feats=img_encoder_outputs[f'l{len(self.img_encoder.layers)-1}'])
        for i, k in enumerate(self.img_decode_guide_map):
            img_decoder_inputs[f'guide{i}'] = img_encoder_outputs[f'{k[0]}']
        img_decoder_outputs = self.img_decoder(img_decoder_inputs)

        depth_encoder_inputs = dict(feats=batch_inputs['sparse_depth'])
        for i, k in enumerate(self.depth_guide_map):
            depth_encoder_inputs[f'guide{i}'] = img_decoder_outputs[f'{k[0]}']
        depth_encoder_ouputs = self.depth_encoder(depth_encoder_inputs)

        decoder_inputs = dict(feats=depth_encoder_ouputs[f'l{len(self.depth_encoder.layers)-1}'] + img_encoder_outputs[f'l{len(self.img_encoder.layers)-1}'])
        for i, k in enumerate(self.decode_guide_map):
            decoder_inputs[f'guide{i}'] = depth_encoder_ouputs[f'{k[0]}']
        decoder_outputs = self.decoder(decoder_inputs)

        depth = self.conv_out(decoder_outputs[f'l{len(self.decoder.layers)-1}']+depth_encoder_ouputs['l0'])
        y = torch.clamp(depth, min=0)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].pred_depth = PixelData(**dict(data=y[i]))
        return batch_data_samples