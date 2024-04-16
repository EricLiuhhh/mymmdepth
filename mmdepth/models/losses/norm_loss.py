import torch
import torch.nn as nn
from mmdepth.registry import MODELS

@MODELS.register_module()
class NormLoss(nn.Module):
    def __init__(self, 
                 p = 1,
                 min_depth = None,
                 max_depth = None,
                 t_valid = 0.0001,
                 normalize_type='mean',
                 ):
        super(NormLoss, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.t_valid = t_valid
        self.normalize_type = normalize_type
        self.p = p
        assert isinstance(self.p, int) and self.p >= 1

    def forward(self, pred, gt):
        if self.min_depth is not None and self.max_depth is not None:
            gt = torch.clamp(gt, min=self.min_depth, max=self.max_depth)
            pred = torch.clamp(pred, min=self.min_depth, max=self.max_depth)
        mask = (gt > self.t_valid).detach()
        if self.normalize_type == 'sep_mean':
            d = torch.abs(pred - gt).pow(self.p) * mask
            d = torch.sum(d, dim=[1, 2, 3])
            num_valid = torch.sum(mask, dim=[1, 2, 3])
            loss = d / (num_valid + 1e-8)
            loss = loss.mean()
        elif self.normalize_type == 'mean':
            loss = (pred-gt)[mask].abs().pow(self.p).mean()
        elif self.normalize_type == 'sum':
            loss = (pred-gt)[mask].abs().pow(self.p).sum()
        else:
            raise NotImplementedError
        return loss

@MODELS.register_module()
class CascadeNormLoss(nn.Module):
    def __init__(self, 
                 p = 1,
                 gamma = 0.8,
                 min_depth = None,
                 max_depth = None,
                 t_valid = 0.0001,
                 normalize_type='mean',
                 ):
        super(CascadeNormLoss, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.t_valid = t_valid
        self.normalize_type = normalize_type
        self.p = p
        self.gamma = gamma
        assert isinstance(self.p, int) and self.p >= 1

    def forward(self, pred, gt):
        loss = 0.0
        n_pred = len(pred)

        if self.min_depth is not None and self.max_depth is not None:
            gt = torch.clamp(gt, min=self.min_depth, max=self.max_depth)
            for i in range(n_pred):
                pred[i] = torch.clamp(pred[i], min=self.min_depth, max=self.max_depth)
        mask = (gt > self.t_valid).detach()

        for i in range(n_pred):
            weight = self.gamma ** (n_pred - i - 1)
            if self.normalize_type == 'sep_mean':
                d = torch.abs(pred[i] - gt).pow(self.p) * mask
                d = torch.sum(d, dim=[1, 2, 3])
                num_valid = torch.sum(mask, dim=[1, 2, 3])
                loss_single = d / (num_valid + 1e-8)
                loss_single = loss_single.mean()
            elif self.normalize_type == 'mean':
                loss_single = (pred[i]-gt)[mask].abs().pow(self.p).mean()
            elif self.normalize_type == 'sum':
                loss_single = (pred[i]-gt)[mask].abs().pow(self.p).sum()
            else:
                raise NotImplementedError
            loss = loss + weight * loss_single
        return loss
