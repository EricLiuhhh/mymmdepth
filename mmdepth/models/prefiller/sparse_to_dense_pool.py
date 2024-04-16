import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdepth.registry import MODELS

@MODELS.register_module()
class SparseToDensePool(BaseModule):
    '''
    Converts sparse inputs to dense outputs using max and min pooling
    with different kernel sizes and combines them with 1 x 1 convolutions

    Arg(s):
        input_channels : int
            number of channels to be fed to max and/or average pool(s)
        min_pool_sizes : list[int]
            list of min pool sizes s (kernel size is s x s)
        max_pool_sizes : list[int]
            list of max pool sizes s (kernel size is s x s)
        n_filter : int
            number of filters for 1 x 1 convolutions
        n_convolution : int
            number of 1 x 1 convolutions to use for balancing detail and density
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''

    def __init__(self,
                 input_channels,
                 min_pool_sizes=[5, 7, 9, 11, 13],
                 max_pool_sizes=[15, 17],
                 n_filter=8,
                 n_convolution=3,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20),
                 norm_cfg=None,
                 init_cfg=None,
                 bias=False):
        super(SparseToDensePool, self).__init__(init_cfg)

        self.min_pool_sizes = [
            s for s in min_pool_sizes if s > 1
        ]

        self.max_pool_sizes = [
            s for s in max_pool_sizes if s > 1
        ]

        # Construct min pools
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.min_pools.append(pool)

        # Construct max pools
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.max_pools.append(pool)

        self.len_pool_sizes = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        pool_convs = []
        for n in range(n_convolution):
            conv = ConvModule(
                in_channels,
                n_filter,
                kernel_size=1,
                stride=1,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                bias=bias)
            pool_convs.append(conv)

            # Set new input channels as output channels
            in_channels = n_filter

        self.pool_convs = torch.nn.Sequential(*pool_convs)

        in_channels = n_filter + input_channels

        self.conv = ConvModule(
            in_channels,
            n_filter,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

    def forward(self, x):
        # Input depth
        z = torch.unsqueeze(x[:, 0, ...], dim=1)

        pool_pyramid = []

        # Use min and max pooling to densify and increase receptive field
        for pool, s in zip(self.min_pools, self.min_pool_sizes):
            # Set flag (999) for any zeros and max pool on -z then revert the values
            z_pool = -pool(torch.where(z == 0, -999 * torch.ones_like(z), -z))
            # Remove any 999 from the results
            z_pool = torch.where(z_pool == 999, torch.zeros_like(z), z_pool)

            pool_pyramid.append(z_pool)

        for pool, s in zip(self.max_pools, self.max_pool_sizes):
            z_pool = pool(z)

            pool_pyramid.append(z_pool)

        # Stack max and minpools into pyramid
        pool_pyramid = torch.cat(pool_pyramid, dim=1)

        # Learn weights for different kernel sizes, and near and far structures
        pool_convs = self.pool_convs(pool_pyramid)

        pool_convs = torch.cat([pool_convs, x], dim=1)

        return self.conv(pool_convs)