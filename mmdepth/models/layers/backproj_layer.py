import torch
import torch.nn as nn
from diagrams import Diagram, Cluster
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
from mmdepth.visualization.diagrams_nodes import *

@MODELS.register_module()
class CalibratedBackprojectionBlock(nn.Module):
    def __init__(self,
                 in_channels_depth,
                 in_channels_fused,
                 out_channels_fused,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20),
                 norm_cfg=None,
                 bias=False):
        super().__init__()
        self.proj_depth = ConvModule(
            in_channels=in_channels_depth,
            out_channels=1,
            kernel_size=1,
            stride=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.conv_fused = ConvModule(
            in_channels=in_channels_fused,
            out_channels=out_channels_fused,
            kernel_size=1,
            stride=2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=bias)
        
    def forward(self, image, depth, coordinates, fused=None):
        layers_fused = []

        # Include image (RGB) features
        layers_fused.append(image)

        # Project depth features to 1 dimension
        z = self.proj_depth(depth)

        # Include backprojected 3D positional (XYZ) encoding: K^-1 [x y 1] z
        xyz = coordinates * z
        layers_fused.append(xyz)

        # Include previous RGBXYZ representation
        if fused is not None:
            layers_fused.append(fused)

        # Obtain fused (RGBXYZ) representation
        layers_fused = torch.cat(layers_fused, dim=1)
        conv_fused = self.conv_fused(layers_fused)

        return conv_fused

@MODELS.register_module()
class CalibratedBackprojectionBlocks(nn.Module):
    '''
    Calibrated backprojection (KB) layer class
    '''
    def __init__(self,
                 in_channels_image=[48, 48, 96, 192],
                 in_channels_depth=[16, 16, 32, 64],
                 out_channels_fused=[48, 96, 192, 384],
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.20)):
        super().__init__()

        self.num_stage = len(out_channels_fused)
        
        blocks = []
        in_channels_fused = 0
        out_fused = 0
        for i in range(self.num_stage):
            in_channels_fused = in_channels_image[i] + out_fused + 3
            blocks.append(CalibratedBackprojectionBlock(in_channels_depth[i], in_channels_fused, out_channels_fused[i], act_cfg=act_cfg, bias=False))
            out_fused = out_channels_fused[i]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        img_feats = inputs['img_feats']
        depth_feats = inputs['depth_feats']
        coords = inputs['coords']
        results = [None]
        for i, (img, depth, coord) in enumerate(zip(img_feats, depth_feats, coords)):
            results.append(self.blocks[i](img, depth, coord, results[-1]))
        return results[1:]
    
    def show_network(self, show=False, filename='net', graph_attr={}, node_attr={}, edge_attr={}, as_cluster=False, cluster_label=None):
        default_graph_attr = {
           "splines":"spline",
        }
        default_node_attr = {
           "imagepos": "tc",
        }
        default_edge_attr = {
        }
        default_graph_attr.update(graph_attr)
        default_node_attr.update(node_attr)
        default_edge_attr.update(edge_attr)
        graph = Diagram if not as_cluster else Cluster
        ext_args = dict(filename=filename, show=show, node_attr=default_node_attr, edge_attr=default_edge_attr) if not as_cluster else dict(label=self.__class__.__name__ if cluster_label is None else cluster_label)
        nodes = dict()
        with graph(graph_attr=default_graph_attr, **ext_args):
            prev = DataNode('inputs')
            for i, block in enumerate(self.blocks):
                block_node = LayerNode(f'l{i}: {block.__class__.__name__}')
                nodes[f'guide_input_l{i}'] = block_node
                prev >> block_node
                prev = block_node
                if i >= 1:
                    hooked_node = DataNode(f'hooked_l{i}')
                    nodes[f'hooked_l{i}'] = hooked_node
                    prev >> hooked_node
        return nodes