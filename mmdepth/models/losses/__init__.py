from .norm_loss import NormLoss, CascadeNormLoss
from .compose_loss import ComposeLoss
from .ssim_loss import SSIMLoss
from .smoothness_loss import EdgeAwareSmoothnessLoss
__all__ = ['NormLoss', 'ComposeLoss', 'CascadeNormLoss', 'SSIMLoss', 'EdgeAwareSmoothnessLoss']