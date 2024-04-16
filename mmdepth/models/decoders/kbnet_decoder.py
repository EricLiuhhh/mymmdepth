import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
from mmdepth.models.encoders import BaseGuidedNet

class _SelfGuide(nn.Module):
    def __init__(self, in_channels, out_channels, act_cfg=None, norm_cfg=None, bias=False, normal_order=True) -> None:
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, 3, 1, 1, act_cfg=act_cfg, norm_cfg=norm_cfg, bias=bias)
        self.normal_order = normal_order

    def forward(self, x, guide):
        guide_conv = self.conv(guide)
        if self.normal_order:
            return torch.cat([x, guide_conv], dim=1)
        else:
            return torch.cat([guide_conv, x], dim=1)

@MODELS.register_module()
class KBNetDecoder(BaseGuidedNet):
    '''
    Multi-scale decoder with skip connections
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_resolution=1,
                 n_filters=[256, 128, 128, 64, 12],
                 n_skips=[256, 128, 64, 32, 0],
                 norm_cfg=None,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20),
                 bias=False,
                 output_func='linear',
                 deconv_type='upconv'):
        super(KBNetDecoder, self).__init__()
        
        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth

        self.n_resolution = n_resolution
        self.output_func = output_func

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2
        
        deconvs = []
        out_convs = []
        guides = []
        self_guide_map = dict()
        guide_cfg = dict(type='CatConvGuide', norm_cfg=None, act_cfg=act_cfg, bias=bias)
        in_channels = input_channels
        self.num_stage = network_depth
        for i in range(network_depth):
            skip_channels, out_channels = n_skips[i], n_filters[i]
            if deconv_type == 'transpose':
                deconv_block = ConvModule(in_channels, out_channels, 3, 2, 1, conv_cfg=dict(type='ConvT', output_padding=1), act_cfg=act_cfg, norm_cfg=norm_cfg, bias=bias)
            elif deconv_type == 'upconv':
                deconv_block = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    ConvModule(in_channels, out_channels, 3, 1, 1, act_cfg=act_cfg, norm_cfg=norm_cfg, bias=bias)
                )
            in_channels = out_channels
            self.loc2channels[f'l{i}'] = in_channels
            deconvs.append(deconv_block)
            if n_resolution > 3 and 3 <= i < network_depth-1:
                out_convs.append(_SelfGuide(out_channels, output_channels, act_cfg, norm_cfg, bias=bias))
                skip_channels += output_channels
                self_guide_map[f'l{i}'] = f'l{i}'
            guides.append(MODELS.build({**guide_cfg, 'feat_planes': out_channels, 'guide_planes': skip_channels}))
        deconvs.append(ConvModule(out_channels, output_channels, 3, 1, 1, act_cfg=None, norm_cfg=None, bias=False))
        self.layers = nn.Sequential(*deconvs)
        self.add_guides(guides, ['l'+str(i) for i in range(self.num_stage)])
        self.add_self_guides(out_convs, self_guide_map)
        self.hook_positions = dict(l=(len(self.layers)-1,))