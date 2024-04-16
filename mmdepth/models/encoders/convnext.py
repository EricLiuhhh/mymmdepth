from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from mmdepth.registry import MODELS
from ..encoders.base_guided_net import BaseGuidedNet

@MODELS.register_module()
class ConvNeXtL(BaseGuidedNet):
    """
    This class utilizes a pre-trained model as a backbone network and
    rearranges it as an encoder for fully convolutional neural network UNet.
    https://github.com/qubvel/segmentation_models.pytorch
    Parameters
    ----------
    backbone: string, default="resnet50"
            Name of classification model (without last dense layers) used as feature extractor to build the model.
            The backbone is created using the timm wrapper (pre-training on ImageNet1K).
    encoder_freeze: boolean, default=False
            If "True" set all layers of encoder (backbone model) as non-trainable.
    pretrained: boolean, default=True
            If true, it download the weights of the pre-trained backbone network in the ImageNet1K dataset.
    preprocessing: boolean, default=False
            If true, the preprocessing step is applied according to the mean and standard deviation of the encoder model.
    non_trainable_layers: tuple, default=(0, 1, 2, 3, 4)
            Specifies which layers are non-trainable.
            For example, if it is 3, the layer of index 3 in the given backbone network are set as non-trainable.
            If the encoder_freeze parameter is set to True, set this non_trainable_layers parameter.
    backbone_kwargs:
            pretrained (bool): load pretrained ImageNet-1k weights if true
            scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
            exportable (bool): set layer config so that model is traceable ONNX exportable (not fully impl/obeyed yet)
            no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)
            drop_rate (float): dropout rate for training (default: 0.0)
            global_pool (str): global pool type (default: 'avg')
            **: other kwargs are model specific for more information;
    backbone_indices: list, default=None
            Out indices, selects which indices to output_images.
    decoder_use_batchnorm: boolean, default=True
            If "True", "BatchNormalisation" layer between "Conv2D" and "Activation" layers is used.
    decoder_channels: tuple, default=(256, 128, 64, 32, 16)
            Tuple of numbers of "Conv2D" layer filters in decoder blocks.
    in_channels: int, default=1
            Specifies the channel number of the image.
    num_classes: int, default=5
            A number of classes for output_images (output_images shape - "(batch, classes, h, w)").
    center: boolean, default=False
            If "True" add "Conv2dReLU" block on encoder head.
    norm_layer: torch.nn.Module, default=nn.BatchNorm2d
            Is a layer that normalizes the activations of the previous layer.
    activation: torch.nn.Module, default=nn.ReLU
            An activation function to apply after the final convolution layer.
            Available options are: **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **relu**.
    """
    def __init__(
            self,
            backbone='convnext_large',
            encoder_freeze=False,
            checkpoint_path=None,
            non_trainable_layers=(0, 1, 2, 3, 4),
            backbone_kwargs=None,
            backbone_indices=None,
            in_channels=3
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        if backbone != "convnext_large":
            raise RuntimeError(
                f"Unknown backbone {backbone}!\n"
                f"Existing model should be convnext_large "
            )

        encoder = create_model(
            backbone, features_only=True, out_indices=backbone_indices,
            in_chans=in_channels, checkpoint_path=checkpoint_path, **backbone_kwargs
        )
        for name, module in encoder.named_children():
            self.layers.add_module(name, module)
        self.hook_positions = dict(l=(2, 3, 4, 5))
        self.loc2channels = dict(zip([f'l{i}' for i in range(2, 6)], [info["num_chs"] for info in encoder.feature_info]))
        self.loc2scales = dict(zip([f'l{i}' for i in range(2, 6)], [1/info["reduction"] for info in encoder.feature_info]))

        if encoder_freeze: self._freeze_encoder(non_trainable_layers)

    def _freeze_encoder(self, non_trainable_layer_idxs):
        """
        Set selected layers non trainable, excluding BatchNormalization layers.
        Parameters
        ----------
        non_trainable_layer_idxs: tuple
            Specifies which layers are non-trainable.
        """
        for idx in non_trainable_layer_idxs:
            for child in self.layers[idx].children():
                for param in child.parameters():
                    param.requires_grad = False
        return

    def from_pretrained(self, weights_path, device='cpu'):
        loaded_dict_enc = torch.load(weights_path, map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.state_dict()}
        self.load_state_dict(filtered_dict_enc)
        self.eval()

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, scale_factor=2.0,
        activation=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, activation=activation)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            if skip!=None:
                x = F.interpolate(x, size=[skip.size(2), skip.size(3)], mode='bilinear', align_corners=True)
            else:
                x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=True,
            activation=nn.ReLU
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(
                channels, channels, scale_factor=1.0,
                activation=activation, norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]

        out_channels = decoder_channels

        if len(in_channels) != len(out_channels):
            in_channels.append(in_channels[-1]//2)

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x

