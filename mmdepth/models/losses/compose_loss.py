import torch
import torch.nn as nn
from mmdepth.registry import MODELS

@MODELS.register_module()
class ComposeLoss(nn.Module):
    def __init__(self,
                 loss_cfgs,
                 loss_weights,
                 loss_names
                 ) -> None:
        super().__init__()

        self.losses = nn.ModuleDict()
        self.weight = []
        for cfg, weight, name in zip(loss_cfgs, loss_weights, loss_names):
            self.losses[name] = MODELS.build(cfg)
            self.weight.append(weight)

    def forward(self, pred, gt):
        losses = {}
        for i, (k, loss_func) in enumerate(self.losses.items()):
            losses[k] = loss_func(pred, gt)
            losses[k] = losses[k] * self.weight[i]
        return losses