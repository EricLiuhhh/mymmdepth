from typing import List, Union, Dict
from mmdepth.registry import MODELS
from .base_completor import BaseCompletor

@MODELS.register_module()
class DualBranchCompletor(BaseCompletor):
    def __init__(self, 
                 img_encoder: Dict,
                 depth_encoder: Dict,
                 fusion_module: Dict,
                 decoder: Dict,
                 data_preprocessor: Dict,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(data_preprocessor, init_cfg)

        self.img_encoder = MODELS.build(img_encoder)
        self.depth_encoder = MODELS.build(depth_encoder)
        self.fusion_module = MODELS.build(fusion_module)
        self.decoder = MODELS.build(decoder)

    def _forward(self, inputs):
        img = inputs['img']
        dep = inputs['sparse_depth']
        img_feat = self.img_encoder(dict(feats=img))
        dep_feat = self.depth_encoder(dict(feats=dep))
        img_feat, dep_feat = self.get_last_feat(img_feat), self.get_last_feat(dep_feat)
        fusion_feat = self.fusion_module(img_feat, dep_feat)
        out = self.decoder(dict(feats=fusion_feat))
        return out
    
    def get_last_feat(self, feat):
        if isinstance(feat, dict):
            return list(feat.values())[-1]
        elif isinstance(feat, (list, tuple)):
            return feat[-1]
        else:
            return feat