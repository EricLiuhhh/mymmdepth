import torch
import torch.nn as nn
import torchvision
from mmdepth.registry import MODELS

@MODELS.register_module()
class RadarNet(nn.Module):
    '''
    Radar association network
    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        n_filters_encoder_image : int
            number of filters for image (RGB) branch
        n_neurons_encoder_depth : int
            number of neurons for depth (radar) branch
        latent_size_depth : int
            size of latent vector
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''
    def __init__(self,
                 input_channels_image=3,
                 input_channels_depth=3,
                 input_patch_size_image=(900, 288),
                 n_filters_encoder_image=[32, 64, 128, 128, 128],
                 n_filters_encoder_depth=[32, 64, 128, 128, 128, 128*29*10],
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20, inplace=True),
                 norm_cfg=None):
        super().__init__()

        self.n_neuron_latent_depth = n_filters_encoder_depth[-2]

        self.encoder_image = MODELS.build(dict(
            type='ResNetRaderNet',
            n_layer=18,
            input_channels=input_channels_image,
            n_filters=n_filters_encoder_image,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        ))

        self.encoder_depth = MODELS.build(dict(
            type='FCEncoder',
            input_channels=input_channels_depth,
            planes=n_filters_encoder_depth,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        ))
        self.input_patch_size_image=input_patch_size_image

    def forward(self, image, points, b_boxes):
        # Image shape: (B, C, H, W) # Should be (B, 3, 768, 288)
        # points shape: (B*K, X)
        # b_boxes: [(K, 4) * B], this should be a list with B elements, and each element is (K, 4) size
        # K is the number of radar points per image
        # X is the radar dimension
        

        # Define dimensions
        shape = self.input_patch_size_image
        latent_height = int(shape[-2] // 32.0)
        latent_width = int(shape[-1] // 32.0)
        batch_size = image.shape[0]

        # Define scales and feature sizes
        skip_scales = [1/2.0, 1/4.0, 1/8.0, 1/16.0, 1/32.0, 1/64.0, 1/128.0]
        skip_feature_sizes = [
            (int(shape[-2] * skip_scale), 
             int(shape[-1] * skip_scale)) 
            for skip_scale in skip_scales
        ] # Should be [(384, 144), (192, 72), (96, 36), (48, 18)]
        
        latent_scale = 1/32.0
        latent_feature_size = (latent_height, latent_width) # Should be (24, 9)

        # Forward the entire image
        latent_image, skips_image = self.encoder_image(image)

        # ROI pooling on latent images
        latent_image_pooled = torchvision.ops.roi_pool(
            latent_image, b_boxes, 
            spatial_scale=latent_scale, 
            output_size=latent_feature_size
        ) # (N*K, C, H, W)
        
        # ROI pooling on the skips
        skips_image_pooled = []
        for skip_image_idx in range(len(skips_image)):
            skips_image_pooled.append(
                torchvision.ops.roi_pool(
                    skips_image[skip_image_idx], b_boxes, 
                    spatial_scale=skip_scales[skip_image_idx], 
                    output_size=skip_feature_sizes[skip_image_idx]
                ) # (N*K, C, H, W)
            )
        
        # Radar points
        # points = points.view(-1, points.shape[-1]) # N, K, X -> N*K, X
        latent_depth = self.encoder_depth(points)
        latent_depth = latent_depth.view(points.shape[0], self.n_neuron_latent_depth, -1, latent_width)
        
        # Concatenate the features
        latent = torch.cat([latent_image_pooled, latent_depth], dim=1)
        return latent, skips_image_pooled