from typing import List, Tuple
import torch
from torch import Tensor
from mmengine.structures import PixelData
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import OptConfigType, OptMultiConfig, OptSampleList, ConfigType, SampleList
from .base_completor import BaseCompletor

@MODELS.register_module()
class KBNet(BaseCompletor):
    def __init__(self,
                 prefiller: ConfigType,
                 img_branch: ConfigType,
                 depth_branch: ConfigType,
                 backproj_layer: ConfigType, 
                 decoder: ConfigType,
                 data_preprocessor: OptConfigType = None, 
                 min_predict_depth = 1.5,
                 max_predict_depth = 100.0,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        self.prefiller = MODELS.build(prefiller)
        self.img_branch = MODELS.build(img_branch)
        self.layer_used_backproj = self.img_branch.backproj_layers
        self.depth_branch = MODELS.build(depth_branch)
        self.backproj_layer = MODELS.build(backproj_layer)
        temp = [a+b for a, b in zip(self.img_branch.n_filters, self.depth_branch.n_filters)]
        decoder['n_skips'] = temp[:-1][::-1]+[0]
        decoder['input_channels'] = temp[-1]
        self.decoder = MODELS.build(decoder)
        self.guide_map[('img_branch', tuple(f'l{i}' for i in self.layer_used_backproj))] = ('backproj_layer', tuple(f'l{i}' for i in range(len(self.layer_used_backproj))))
        self.guide_map[('depth_branch', tuple(f'l{i}' for i in self.layer_used_backproj))] = ('backproj_layer', tuple(f'l{i}' for i in range(len(self.layer_used_backproj))))
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        pass

    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        pass

    def predict(self, inputs: Tensor, batch_data_samples: OptSampleList = None, mode: str = 'tensor'):
        prefilled_depth = self.prefiller(torch.cat([inputs['sparse_depth'], inputs['valid_map']], dim=1))
        img_outputs = self.img_branch(dict(feats=inputs['img']), stop_at=len(self.layer_used_backproj)+1)

        depth_branch_inputs = dict(feats=prefilled_depth)
        for i in range(len(self.layer_used_backproj)):
            depth_branch_inputs.update({f'guide{i}': inputs['pos_embds'][i]})
        depth_outputs = self.depth_branch(depth_branch_inputs)

        backproj_layer_inputs = dict(
            img_feats=[img_outputs[f'l{i}'] for i in self.layer_used_backproj],
            depth_feats=[depth_outputs[f'l{i}'] for i in self.layer_used_backproj],
            coords=inputs['pos_embds'])
        conv_fused = self.backproj_layer(backproj_layer_inputs)

        img_outputs_2 = self.img_branch(dict(feats=conv_fused[-1]))
        img_outputs.update(img_outputs_2)
        
        num_stage = len(img_outputs)
        decoder_inputs = dict()
        flag = 0 in self.layer_used_backproj
        temp = num_stage-1 if flag else num_stage
        for i in range(temp):
            key = 'feats' if i == temp-1 else f'guide{temp-i-2}'
            j = i+1 if flag else i  # extra conv layer
            if i in self.layer_used_backproj:
                decoder_inputs.update({key: torch.cat([conv_fused[self.layer_used_backproj[i]], depth_outputs[f'l{j}']], dim=1)})
            else:
                decoder_inputs.update({key: torch.cat([img_outputs[f'l{j}'], depth_outputs[f'l{j}']], dim=1)})
        decoder_inputs[f'guide{temp-1}'] = None
        decoder_outputs = self.decoder(decoder_inputs)
        output_depth = torch.sigmoid(decoder_outputs[f'l{len(self.decoder.layers)-1}'])
        output_depth = \
            self.min_predict_depth / (output_depth + self.min_predict_depth / self.max_predict_depth)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].pred_depth = PixelData(**dict(data=output_depth[i]))
        return batch_data_samples