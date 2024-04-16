from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import copy
import torch
from torch import Tensor
from diagrams import Diagram
from mmengine.model import BaseModel
from mmengine.structures import PixelData
from mmdepth.utils.typing_utils import OptConfigType, OptMultiConfig, SampleList, OptSampleList

class BaseCompletor(BaseModel, metaclass=ABCMeta):
    """Base class for completors.

    Args:
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_val``,
            ``mean`` and ``std``. Default to None.
       init_cfg (dict, optional): the config to control the
           initialization. Default to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.guide_map = dict()
        
    def build_guide(self, guide, provider, receiver):
        guide_map = guide['guide_map']
        guide_cfg = guide['guide_cfg']
        if not isinstance(guide_cfg, (list, tuple)):
            guide_cfg = [copy.deepcopy(guide_cfg) for _ in range(len(guide_map))]
        assert len(guide_cfg) == len(guide_map), 'len(guide_cfg) != len(guide_map)'
        for i in range(len(guide_cfg)):
            guide_cfg[i]['guide_planes'] = provider.get_planes(loc=guide_map[i][0])
            guide_cfg[i]['feat_planes'] = receiver.get_planes(loc=guide_map[i][-1])
        _, t = zip(*guide_map)
        receiver.add_guides(guide_cfg, t)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_depths = [
            data_sample.gt_depth.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_depths, dim=0)
        
    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    #@abstractmethod
    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        raise NotImplementedError

    def show_network(self, show=False, filename='net', graph_attr={}, node_attr={}, edge_attr={}):
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
        members = self._modules
        nodes = dict()
        with Diagram(filename=filename, show=show, graph_attr=default_graph_attr, node_attr=default_node_attr, edge_attr=default_edge_attr):
            for k, v in members.items():
                if hasattr(v, 'show_network'):
                    nodes[k] = v.show_network(show=False, graph_attr=default_graph_attr, node_attr=default_node_attr, edge_attr=default_edge_attr, as_cluster=True, cluster_label=k)
            for s, t in self.guide_map.items():
                source_name, source_hooks = s[0], s[1]
                target_name, target_hooks = t[0], t[1]
                assert len(source_hooks) == len(target_hooks)
                for sh, th in zip(source_hooks, target_hooks):
                    nodes[source_name][f'hooked_{sh}'] >> nodes[target_name][f'guide_input_{th}']
        return