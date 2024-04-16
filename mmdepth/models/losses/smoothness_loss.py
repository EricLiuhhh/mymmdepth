import torch
import torch.nn as nn
from mmdepth.registry import MODELS

@MODELS.register_module()
class EdgeAwareSmoothnessLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img, pred):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_pred_x *= torch.exp(-grad_img_x)
        grad_pred_y *= torch.exp(-grad_img_y)

        return grad_pred_x.mean() + grad_pred_y.mean()