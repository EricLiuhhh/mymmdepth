from typing import List, Union, Optional
import torch.nn as nn
from diagrams import Diagram, Cluster

from mmengine.model import BaseModule
from mmengine.logging import print_log
from mmdepth.utils.typing_utils import ConfigType
from mmdepth.registry import MODELS
from mmdepth.visualization.diagrams_nodes import *

@MODELS.register_module()
class BaseGuidedNet(BaseModule):
    '''
    ->layer1->guide1->layer2->guide2->layer3->...
                |
            inputs['guide1']
    '''
    def __init__(self,
                 guides: Optional[List[Union[nn.Module, ConfigType]]] = None,
                 guide_locations: Optional[List[str]] = None,
                 hook_positions: Optional[List[str]] = dict(l=(), g=()),
                 self_guides: Optional[List[Union[nn.Module, ConfigType]]] = None,
                 self_guide_map: dict = {},
                 hook_before_self_guide = False,
                 init_cfg = None):
        super().__init__(init_cfg)
        self.loc2channels = dict()
        self.loc2scales = dict()
        self.layers = nn.Sequential()
        if guides is not None:
            assert len(guide_locations) == len(guides)
            self.add_guides(guides, guide_locations)
        else:
            self.guide_locations = []
        if self_guides is not None:
            assert len(self_guide_map) == len(self_guides)
            self.add_self_guides(self_guides, self_guide_map)
        else:
            self.self_guide_map = self_guide_map
            self.self_guides = [(lambda x, y: x + y) for _ in range(len(self.self_guide_map))]
        self.hook_positions = hook_positions
        self.hook_before_self_guide = hook_before_self_guide
        self.status_cache = dict()

    def add_guides(self, 
                   guides: List[Union[nn.Module, ConfigType]], 
                   guide_locations: List[str]):
        self.guides = self._build_blocks(guides)
        for i in range(len(self.guides)):
            if hasattr(self.guides[i], 'out_channels'):
                self.loc2channels[f'g{i}'] = self.guides[i].out_channels
            elif isinstance(guides[i], ConfigType) and guides[i].get('out_channels', False):
                self.loc2channels[f'g{i}'] = guides[i].get('out_channels')
            else:
                raise RuntimeError('Can not inference output channels of Guide Module.')
            self.loc2scales[f'g{i}'] = self.guides[i].scale * self.loc2scales[guide_locations[i]] if hasattr(self.guides[i], 'scale') else self.loc2scales[guide_locations[i]]

        self.guide_locations = guide_locations

    def add_self_guides(self, 
                   guides: List[Union[nn.Module, ConfigType]], 
                   self_guide_map: dict):
        self.self_guides = self._build_blocks(guides)
        self.self_guide_map = self_guide_map

    def _build_blocks(self, blocks: Union[List[nn.Module], List[ConfigType]]):
        results = []
        for block in blocks:
            if isinstance(block, nn.Module):
                results.append(block)
            elif isinstance(block, dict):
                results.append(MODELS.build(block))
            else:
                raise NotImplementedError
        return nn.Sequential(*results)
    
    def get_planes(self, loc:Union[str, int]):
        '''
        loc: [l, g][0, 9]
        '''
        if isinstance(loc, str):
            return self.loc2channels[loc]
        elif isinstance(loc, int):
            if loc >= 0:
                return self.loc2channels[f'l{loc}']
            else:
                return self.loc2channels[f'l{len(self.layers)+loc}']
            
    def get_scale(self, loc:Union[str, int]):
        '''
        loc: [l, g][0, 9]
        '''
        if isinstance(loc, str):
            return self.loc2scales[loc]
        elif isinstance(loc, int):
            if loc >= 0:
                return self.loc2scales[f'l{loc}']
            else:
                return self.loc2scales[f'l{len(self.layers)+loc}']

    def forward(self, inputs, stop_at=None):
        x = inputs['feats']
        self_guide_feats = {}
        results = {}
        loc_idx = 0 if 'loc_idx' not in self.status_cache else self.status_cache['loc_idx']
        self_guide_idx = 0 if 'self_guide_idx' not in self.status_cache else self.status_cache['self_guide_idx']
        num_stage = len(self.layers) if stop_at is None else stop_at
        start_stage = 0 if 'i' not in self.status_cache else self.status_cache['i']
        for i in range(start_stage, num_stage):
            if f'ext_feats_l{i}' in inputs:
                if isinstance(inputs[f'ext_feats_l{i}'], (list, tuple)):
                    x = self.layers[i](x, *inputs[f'ext_feats_l{i}'])
                else:
                    x = self.layers[i](x, inputs[f'ext_feats_l{i}'])
            else:
                x = self.layers[i](x)
            
            if self.hook_before_self_guide:
                if 'l' in self.hook_positions and i in self.hook_positions['l']: 
                    results[f'l{i}'] = x
            if f'l{i}' in self.self_guide_map:
                self_guide_feats[self.self_guide_map[f'l{i}']] = x
            if f'l{i}' in self.self_guide_map.values():
                #x = x + self_guide_feats[f'l{i}']
                x = self.self_guides[self_guide_idx](x, self_guide_feats[f'l{i}'])
                self_guide_idx += 1
            if not self.hook_before_self_guide:
                if 'l' in self.hook_positions and i in self.hook_positions['l']: 
                    results[f'l{i}'] = x

            if f'l{i}' in self.guide_locations:
                if isinstance(inputs[f'guide{loc_idx}'], dict):
                    x = self.guides[loc_idx](x, **inputs[f'guide{loc_idx}'])
                else:
                    x = self.guides[loc_idx](x, inputs[f'guide{loc_idx}'])
                if 'g' in self.hook_positions and loc_idx in self.hook_positions['g']:
                    results[f'g{loc_idx}'] = x
                loc_idx += 1
            
            if i == len(self.layers)-1:
                self.status_cache = dict()
            elif i == num_stage-1:
                self.status_cache = dict(i=num_stage, loc_idx=loc_idx, self_guide_idx=self_guide_idx)
        return results
    
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
            if len(self.layers) == 0:
                print_log('len(self.layers) == 0, nothing to show', 'current', 'WARNING')
                return
            inputs = DataNode('Inputs')
            nodes['inputs'] = inputs
            prev = inputs
            guide_idx = 0
            self_guide_idx = 0
            self_guide_node_map = dict()
            for i in range(len(self.layers)):
                layer_node = LayerNode(f'l{i}: {self.layers[i].__class__.__name__}')
                nodes[f'l{i}'] = layer_node
                prev >> layer_node
                prev = layer_node
                if self.hook_before_self_guide:
                    if 'l' in self.hook_positions and i in self.hook_positions['l']: 
                        hooked_node = HookNode(f'hooked_l{i}')
                        nodes[f'hooked_l{i}'] = hooked_node
                        prev >> hooked_node
                if f'l{i}' in self.self_guide_map:
                    self_guide_node_map[self.self_guide_map[f'l{i}']] = layer_node
                if f'l{i}' in self.self_guide_map.values():
                    guide_input = self_guide_node_map[f'l{i}']
                    guide_node = SelfGuideNode(f'self_guide{self_guide_idx}: {self.self_guides[self_guide_idx].__class__.__name__}')
                    nodes[f'self_guide{self_guide_idx}'] = guide_node
                    [guide_input, prev] >> guide_node
                    prev = guide_node
                    self_guide_idx += 1
                if not self.hook_before_self_guide:
                    if 'l' in self.hook_positions and i in self.hook_positions['l']: 
                        hooked_node = HookNode(f'hooked_l{i}')
                        nodes[f'hooked_l{i}'] = hooked_node
                        prev >> hooked_node
                if f'l{i}' in self.guide_locations:
                    ext_guide = DataNode(f'guide_input_l{i}')
                    nodes[f'guide_input_l{i}'] = ext_guide
                    guide_node = GuideNode(f'guide{guide_idx}: {self.guides[guide_idx].__class__.__name__}')
                    nodes[f'guide{guide_idx}'] = guide_node
                    [prev, ext_guide] >> guide_node
                    prev = guide_node
                    if 'g' in self.hook_positions and guide_idx in self.hook_positions['g']:
                        hooked_node = HookNode(f'hooked_g{guide_idx}')
                        nodes[f'hooked_g{guide_idx}'] = hooked_node
                        prev >> hooked_node
                    guide_idx += 1
        return nodes