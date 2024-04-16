import sys
sys.path.append('.')
import torch
# from mmdepth.models.encoders import ConvNeXtL
# m = ConvNeXtL()
# m(dict(feats=torch.randn((1, 3, 224, 224))))

from SQLdepth.src.models.depth_encoders.convnext.convnextL_encoder import ConvNeXtLEncoderDecoder

m=ConvNeXtLEncoderDecoder()
pass