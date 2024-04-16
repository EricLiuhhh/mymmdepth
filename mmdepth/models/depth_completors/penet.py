from typing import List, Tuple
import torch
from mmengine.structures import PixelData
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import OptConfigType, OptMultiConfig, ConfigType, OptSampleList, SampleList
from .base_completor import BaseCompletor

@MODELS.register_module()
class PENet(BaseCompletor):
    def __init__(self, 
                 img_branch: ConfigType,
                 depth_guide: ConfigType,
                 depth_branch: ConfigType,
                 refinement: ConfigType,
                 data_preprocessor: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        img_branch.update(geoplanes=self.data_preprocessor.geoplanes)
        self.img_branch = MODELS.build(img_branch)
        depth_branch.update(geoplanes=self.img_branch.geoplanes)
        self.depth_branch = MODELS.build(depth_branch)
        self.depth_guide_map = depth_guide['guide_map']
        self.build_guide(depth_guide, self.img_branch, self.depth_branch)
        self.refinement = MODELS.build(refinement)

    def loss(self, batch_inputs, batch_data_samples):
        coarse = self.backbone(batch_inputs)
        refine = self.refinement(coarse)
    
    def _forward(self, batch_inputs, batch_data_samples=None):
        coarse = self.backbone(batch_inputs)
        refine = self.refinement(coarse)
        return refine
    
    def predict(self, batch_inputs, batch_data_samples=None) -> SampleList:
        img_branch_inputs = dict(feats=torch.cat((batch_inputs['img'], batch_inputs['sparse_depth']), dim=1), **batch_inputs)
        img_branch_outputs = self.img_branch(img_branch_inputs)

        rgb_depth, rgb_conf = torch.chunk(img_branch_outputs['l11'], 2, dim=1)
        depth_encoder_inputs = dict(feats = torch.cat((batch_inputs['sparse_depth'], rgb_depth), dim=1), **batch_inputs)
        for i, k in enumerate(self.depth_guide_map):
            depth_encoder_inputs[f'guide{i}'] = img_branch_outputs[f'{k[0]}']
        depth_encoder_outputs = self.depth_branch(depth_encoder_inputs)
        
        d_depth, d_conf = torch.chunk(depth_encoder_outputs['l11'], 2, dim=1)
        rgb_conf, d_conf = torch.chunk(torch.softmax(torch.cat((rgb_conf, d_conf), dim=1), dim=1), 2, dim=1)
        coarse_depth = rgb_conf*rgb_depth + d_conf*d_depth
        
        refinement_inputs = dict(sparse_depth=batch_inputs['sparse_depth'], coarse_depth=coarse_depth, guide0=torch.cat((img_branch_outputs['l9'], depth_encoder_outputs['l9']), 1), guide1=torch.cat((img_branch_outputs['l10'], depth_encoder_outputs['l10']), 1))
        refined_depth = self.refinement(refinement_inputs)

        y = torch.clamp(refined_depth, min=0)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].pred_depth = PixelData(**dict(data=y[i]))
        return batch_data_samples