from typing import List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule, build_upsample_layer
from mmdepth.registry import MODELS
from mmdepth.models.data_preprocessors.penet_data_preprocessor import SparseDownSampleClose

class CSPNGenerateAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        norm_cfg = dict(type='BN')
        self.generate = ConvModule(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1, act_cfg=None, norm_cfg=norm_cfg)

    def forward(self, feature):

        guide = self.generate(feature)

        #normalization in standard CSPN
        #'''
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        #'''
        #weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output =  torch.cat((half1, guide_mid, half2), dim=1)
        return output
    
class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, kernel, input, input0): #with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size*self.kernel_size-1)/2)
        input_im2col[:, mid_index:mid_index+1, :] = input0

        #print(input_im2col.size(), kernel.size())
        output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel))
        return output.view(bs, 1, h, w)
    
    def __repr__(self):
        return f'CSPNAccelerate(kernel_size={self.kernel_size}, dilation={self.dilation}, padding={self.padding}, stride={self.stride})'

def kernel_trans(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size-1)/2))
    return kernel

def gen_trans_kernel(kernel_size):
    ks = kernel_size
    encoder = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
    kernel_range_list = [i for i in range(ks - 1, -1, -1)]
    ls = []
    for i in range(ks):
        ls.extend(kernel_range_list)
    index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                [val for val in kernel_range_list for j in range(ks)], ls]
    encoder[index] = 1
    return nn.Parameter(encoder, requires_grad=False)

@MODELS.register_module()
class DACSPNPP(BaseModule):
    def __init__(self, 
                 upsample_type='nearest',
                 in_channels=(128, 64),
                 init_cfg=None):
        super().__init__(init_cfg)
        norm_cfg = dict(type='BN')

        self.num_stage = len(in_channels)
        kernel_conf_layers = []
        depth_conf_layers = []
        affinity_layers = []
        cspn_layers = []
        upsample_layers = []
        for i in range(self.num_stage):
            kernel_conf_layers.append(ConvModule(in_channels[i], 3, kernel_size=3, stride=1, padding=1, act_cfg=None, norm_cfg=norm_cfg))
            depth_conf_layers.append(ConvModule(in_channels[i], 1, kernel_size=3, stride=1, padding=1, act_cfg=None, norm_cfg=norm_cfg))
            affinity_layers.append(nn.Sequential(
                CSPNGenerateAccelerate(in_channels[i], 3),
                CSPNGenerateAccelerate(in_channels[i], 5),
                CSPNGenerateAccelerate(in_channels[i], 7)
            )) 
            upsample_scale = 2**(self.num_stage-i-1)
            cspn_layers.append(nn.Sequential(
                CSPNAccelerate(kernel_size=3, dilation=upsample_scale, padding=1*upsample_scale, stride=1),
                CSPNAccelerate(kernel_size=5, dilation=upsample_scale, padding=2*upsample_scale, stride=1),
                CSPNAccelerate(kernel_size=7, dilation=upsample_scale, padding=3*upsample_scale, stride=1)
            ))
            if upsample_scale == 1:
                upsample_layers.append(nn.Sequential())
            else:
                upsample_layers.append(build_upsample_layer(dict(type=upsample_type, scale_factor=upsample_scale)))

        self.kernel_conf_layers = nn.Sequential(*kernel_conf_layers)
        self.depth_conf_layers = nn.Sequential(*depth_conf_layers)
        self.affinity_layers = nn.Sequential(*affinity_layers)
        self.cspn_layers = nn.Sequential(*cspn_layers)
        self.upsample_layers = nn.Sequential(*upsample_layers)

        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)

        self.trans_kernels = []
        for k in [3, 5, 7]:
            self.trans_kernels.append(gen_trans_kernel(k))
        self.trans_kernels = nn.ParameterList(self.trans_kernels)

    def forward(self, inputs):
        coarse_depth = inputs['coarse_depth']
        d = inputs['sparse_depth']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        ds = [d]
        valid_masks = [valid_mask]
        for i in range(self.num_stage-1):
            temp = self.downsample(ds[-1], valid_masks[-1])
            ds.append(temp[0])
            valid_masks.append(temp[1])

        depths = [coarse_depth, coarse_depth, coarse_depth]
        refined_depth = coarse_depth
        for i in range(self.num_stage):
            guide = inputs[f'guide{i}']
            depth_conf = self.depth_conf_layers[i](guide)
            depth_conf = torch.sigmoid(depth_conf) * valid_masks[-i-1]
            kernel_conf = self.kernel_conf_layers[i](guide)
            kernel_conf = self.softmax(kernel_conf)
            kernel_conf_up = []
            affinity_up = []
            for k in range(3):
                kernel_conf_up.append(self.upsample_layers[i](kernel_conf[:, k:k+1, :, :]))
                affinity_up.append(self.upsample_layers[i](kernel_trans(self.affinity_layers[i][k](guide), self.trans_kernels[k])))
            depth_up = self.upsample_layers[i](ds[-i-1])
            depth_conf_up = self.upsample_layers[i](depth_conf)
            for _ in range(6):
                for k in range(3):
                    depths[k] = self.cspn_layers[i][k](affinity_up[k], depths[k], refined_depth)
                    depths[k] = depth_conf_up*depth_up + (1-depth_conf_up)*depths[k]
            refined_depth = kernel_conf_up[0]*depths[0]+kernel_conf_up[1]*depths[1]+kernel_conf_up[2]*depths[2]
            depths = [refined_depth, refined_depth, refined_depth]

        return refined_depth