from typing import List, Tuple
from torch import Tensor
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig
from .base_completor import BaseCompletor

@MODELS.register_module()
class RadarNet(BaseCompletor):
    def __init__(self,
                 prefiller: ConfigType,
                 fusion_net: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        self.prefiller = MODELS.build(prefiller)
        self.fusion_net = MODELS.build(fusion_net)

    def _forward(self, inputs: Tensor, data_samples = None) -> Tuple[List[Tensor]]:
        x = self.prefiller(inputs)