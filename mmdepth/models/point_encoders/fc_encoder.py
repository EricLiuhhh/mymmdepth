import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdepth.registry import MODELS
from ..encoders import BaseGuidedNet

@MODELS.register_module()
class FCModule(nn.Module):
    '''
    Fully connected layer

    Arg(s):
        in_channels : int
            number of input neurons
        out_channels : int
            number of output neurons
        dropout_rate : float
            probability to use dropout
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 dropout_rate=0.00,
                 init_cfg=None):
        super().__init__()

        self.fc = torch.nn.Linear(in_channels, out_channels)

        if act_cfg is not None:
            self.act_func = build_activation_layer(act_cfg)
        else:
            self.act_func = None

        if norm_cfg is not None:
            self.norm_layer = build_norm_layer(norm_cfg)
        else:
            self.norm_layer = None

        if dropout_rate > 0.00 and dropout_rate <= 1.00:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.fc(x)

        if self.norm_layer is not None:
            out = self.norm_layer(out)

        if self.act_func is not None:
            out = self.act_func(out)

        if self.dropout is not None:
            return self.dropout(out)
        else:
            return out

@MODELS.register_module()
class FCEncoder(BaseGuidedNet):
    '''
    Fully connected encoder
    Arg(s):
        input_channels : int
            number of input channels
        n_neurons : list[int]
            number of filters to use per layer
        latent_size : int
            number of output neuron
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
    '''

    def __init__(self,
                 input_channels,
                 planes,
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 dropout_ratio=0,
                 hook_positions='last'):
        super().__init__()
        
        layers = []
        num_stage = len(planes)
        in_channels = input_channels
        for i in range(num_stage):
            out_channels = planes[i]
            layers.append(FCModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        dropout_rate=dropout_ratio))
            in_channels = out_channels
            self.loc2scales[f'l{i}'] = out_channels
        self.layers = nn.Sequential(layers)
        if self.hook_positions == 'last':
            self.hook_positions = f'l{num_stage-1}'
        else:
            self.hook_positions = hook_positions