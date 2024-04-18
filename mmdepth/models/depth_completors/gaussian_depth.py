from typing import List, Tuple, Sequence
import torch
from torch import Tensor
import torch.nn as nn
from mmengine.config import ConfigDict
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import SampleList, DepthDataSample, ConfigType
from .base_completor import BaseCompletor

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

@MODELS.register_module()
class GaussianDepth(BaseCompletor):
    def __init__(self, 
                 point_encoder: ConfigType,
                 # img_encoder: ConfigType,
                 prefiller: ConfigType,
                 data_preprocessor = None, 
                 init_cfg = None):
        super().__init__(data_preprocessor, init_cfg)
        self.prefiller = MODELS.build(prefiller)
        self.point_encoder = MODELS.build(point_encoder)
        # self.img_encoder = MODELS.build(img_encoder)
        self.gaussian_proj = nn.Linear(64, 7) # (rot scale opa)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
    def _forward(self, inputs: Tensor, data_samples = None) -> Tuple[List[Tensor]]:
        pass
    
    def predict(self, inputs: Tensor, data_samples = None) -> Sequence[DepthDataSample]:
        pass

    def loss(self, inputs: Tensor, data_samples: Sequence[DepthDataSample]) -> dict:
        points_feat = self.extract_points_feat(inputs)
        gaussian_params = self.gaussian_proj(points_feat)
        rot = self.rotation_activation(gaussian_params[:, :3])
        scale = self.scaling_activation(gaussian_params[:, 3:6])
        opa = self.opacity_activation(gaussian_params[:, 6:6])
        xyz = inputs['voxels']['voxels'][:, :3]
        coords = inputs['voxels']['coords']
        self.prefiller(inputs, data_samples)

    def extract_points_feat(self, batch_inputs_dict: dict) -> Tensor:
        """Extract features from voxels.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.

        Returns:
            SparseTensor: voxels with features.
        """
        voxel_dict = batch_inputs_dict['voxels']
        x = self.point_encoder(voxel_dict['voxels'], voxel_dict['coors'])
        return x