import torch.nn as nn
from torch.autograd import Function
from mmdepth.exts import GuideConv
from mmdepth.registry import MODELS

class Conv2dLocal_F(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = GuideConv.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = GuideConv.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight

@MODELS.register_module()
class Conv2dGuided(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        input, weight = inputs
        output = Conv2dLocal_F.apply(input, weight)
        return output