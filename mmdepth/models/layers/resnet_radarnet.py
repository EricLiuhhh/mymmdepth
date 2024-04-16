import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, ConvModule
from mmdepth.registry import MODELS

@MODELS.register_module()
class ResNetBlockRadarNet(nn.Module):
    '''
    RadarNet uses slightly different structrue comperad with original ResNet block.
    Basic ResNet block class
    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20, inplace=True),
                 norm_cfg=None):
        super().__init__()

        self.activation_func = build_activation_layer(act_cfg)

        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=False)

        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=False)

        self.projection = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            act_cfg=None,
            norm_cfg=norm_cfg,
            bias=False)

    def forward(self, x):
        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)

@MODELS.register_module()
class ResNetRadarNet(nn.Module):
    '''
    ResNet encoder with skip connections
    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20, inplace=True),
                 norm_cfg=None):
        super().__init__()

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = ResNetBlockRadarNet
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = ResNetBlockRadarNet
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=7//2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks2 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks3 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks4 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks5 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.blocks6 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.blocks7 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)
        else:
            self.blocks7 = None

    def _make_layer(self,
                    network_block,
                    n_block,
                    in_channels,
                    out_channels,
                    stride,
                    act_cfg,
                    norm_cfg):
        '''
        Creates a layer
        Arg(s):
            network_block : Object
                block type
            n_block : int
                number of blocks to use in layer
            in_channels : int
                number of channels
            out_channels : int
                number of output channels
            stride : int
                stride of convolution
            weight_initializer : str
                kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
            activation_func : func
                activation function after convolution
            use_batch_norm : bool
                if set, then applied batch normalization
        '''

        blocks = []

        for n in range(n_block):

            if n == 0:
                stride = stride
            else:
                in_channels = out_channels
                stride = 1

            block = network_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)

            blocks.append(block)

        blocks = torch.nn.Sequential(*blocks)

        return blocks

    def forward(self, x):
        '''
        Forward input x through the ResNet model
        Arg(s):
            x : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]