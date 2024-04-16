'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdepth.registry import MODELS
from mmdepth.models.encoders.resnet import ResNet
from mmdepth.models.utils import pose_utils

@MODELS.register_module()
class PoseEncoder(torch.nn.Module):
    '''
    Pose network encoder

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list[int]
            number of filters to use for each convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 input_channels=6,
                 n_filters=[16, 32, 64, 128, 256, 256, 256],
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 norm_cfg=None):
        super(PoseEncoder, self).__init__()

        self.conv1 = ConvModule(
            input_channels,
            n_filters[0],
            kernel_size=7,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.conv2 = ConvModule(
            n_filters[0],
            n_filters[1],
            kernel_size=5,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.conv3 = ConvModule(
            n_filters[1],
            n_filters[2],
            kernel_size=3,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.conv4 = ConvModule(
            n_filters[2],
            n_filters[3],
            kernel_size=3,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.conv5 = ConvModule(
            n_filters[3],
            n_filters[4],
            kernel_size=3,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.conv6 = ConvModule(
            n_filters[4],
            n_filters[5],
            kernel_size=3,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.conv7 = ConvModule(
            n_filters[5],
            n_filters[6],
            kernel_size=3,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        '''
        Forward input x through encoder

        Arg(s):
            x : torch.Tensor[float32]
                input image N x C x H x W
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            None
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        # Resolution 1/32 -> 1/64
        layers.append(self.conv6(layers[-1]))

        # Resolution 1/64 -> 1/128
        layers.append(self.conv7(layers[-1]))

        return layers[-1], None

@MODELS.register_module()
class PoseDecoder(torch.nn.Module):
    '''
    Pose Decoder 6 DOF

    Arg(s):
        rotation_parameterization : str
            axis
        input_channels : int
            number of channels in input latent vector
        n_filters : int list
            number of filters to use at each decoder block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 rotation_parameterization='axis',
                 input_channels=256,
                 n_filters=[],
                 squeeze_channels=None,
                 stride=2,
                 act_out=True,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=None):
        super(PoseDecoder, self).__init__()

        self.rotation_parameterization = rotation_parameterization

        if len(n_filters) > 0:
            layers = []
            in_channels = input_channels

            if squeeze_channels is not None:
                conv = ConvModule(in_channels, squeeze_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
                layers.append(conv)
                in_channels = squeeze_channels

            for out_channels in n_filters:
                conv = ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg)
                layers.append(conv)
                in_channels = out_channels

            conv = ConvModule(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                act_cfg=act_cfg if act_out else None,
                norm_cfg=norm_cfg)
            layers.append(conv)

            self.conv = torch.nn.Sequential(*layers)
        else:
            self.conv = ConvModule(
                in_channels=input_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)

    def forward(self, x, invert=False):
        conv_output = self.conv(x)
        pose_mean = torch.mean(conv_output, [2, 3])
        dof = 0.01 * pose_mean
        posemat = pose_utils.pose_matrix(
            dof,
            rotation_parameterization=self.rotation_parameterization, invert=invert)

        return posemat

@MODELS.register_module()
class PoseNet(BaseModule):
    '''
    Pose network for computing relative pose between a pair of images

    Arg(s):
        encoder_type : str
            posenet, resnet18, resnet34
        rotation_parameterization : str
            rotation parameterization: axis
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type='posenet',
                 rotation_parameterization='axis',
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg)

        # Create pose encoder
        if encoder_type == 'posenet':
            self.encoder = PoseEncoder(
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256, 256, 256],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)
        elif encoder_type == 'resnet18':
            self.encoder = ResNet(depth=18, 
                                  in_channels=6,
                                  base_channels=16)
        elif encoder_type == 'resnet34':
            self.encoder = ResNet(depth=34, 
                                  in_channels=6,
                                  base_channels=16)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))

        # Create pose decoder
        if encoder_type == 'posenet':
            self.decoder = PoseDecoder(
                rotation_parameterization=rotation_parameterization,
                input_channels=256)
        elif encoder_type == 'resnet18' or encoder_type == 'resnet34':
            self.decoder = PoseDecoder(
                rotation_parameterization=rotation_parameterization,
                input_channels=256,
                n_filters=[256, 256],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))


    def forward(self, image0, image1):
        '''
        Forwards the inputs through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                image at time step 0
            image1 : torch.Tensor[float32]
                image at time step 1
        Returns:
            torch.Tensor[float32] : pose from time step 1 to 0
        '''

        # Forward through the network
        latent, _ = self.encoder(torch.cat([image0, image1], dim=1))
        output = self.decoder(latent)

        return output
