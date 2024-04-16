import torch.nn as nn
from mmdepth.registry import MODELS
from .data_preprocessors import *
from .depth_completors import *
from .depth_estimators import *
from .encoder_decoder import *
from .encoders import *
from .decoders import *
from .posenet import *
from .refinements import *
from .layers import *
from .prefiller import *
from .losses import *
from .point_encoders import *
from .radar_prefiller import *

MODELS.register_module('ConvT', module=nn.ConvTranspose2d)
MODELS.register_module('Hardswish', module=nn.Hardswish)