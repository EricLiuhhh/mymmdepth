import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS

class _CAM(nn.Module):
    def __init__(self, high_feature_channel, output_channel = None):
        super(_CAM, self).__init__()
        in_channel = high_feature_channel 
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = MODELS.build(dict(type='ChannelAttention', in_planes=channel, pool_type=['avg']))
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features):
        features = high_features
        features = self.ca(features) * features        
        return self.relu(self.conv_se(features))
    
@MODELS.register_module()
class HRDepthDecoder(nn.Module):
    def __init__(self, 
                 num_ch_enc, 
                 use_input_attention=False,
                 input_channels=None,
                 scales=range(4), 
                 num_output_channels=1, 
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 padding_mode='zeros',
                 mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.mobile_encoder = mobile_encoder
        if mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.use_input_attention = use_input_attention
        if use_input_attention:
            assert (input_channels is not None) and (len(input_channels) == len(num_ch_enc))
            self.input_attentions = []
            for i in range(len(input_channels)-1, 0, -1):
                self.input_attentions.append(_CAM(input_channels[i], num_ch_enc[i]))
            self.input_attentions.append(nn.Identity())
            self.input_attentions = nn.Sequential(*self.input_attentions)

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in = num_ch_in // 2
                num_ch_out = num_ch_in // 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvModule(num_ch_in, num_ch_out, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvModule(num_ch_in, num_ch_out, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_" + index + "_attention"] = MODELS.build(dict(type='fSEModule', high_feature_channel=num_ch_enc[row + 1] // 2, low_feature_channels=self.num_ch_enc[row] + self.num_ch_dec[row]*2*(col-1), output_channel=self.num_ch_dec[row] * 2))
            else:
                self.convs["X_" + index + "_attention"] = MODELS.build(dict(type='fSEModule', high_feature_channel=num_ch_enc[row + 1] // 2, low_feature_channels=self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1)))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvModule(
                    self.num_ch_enc[row]+ self.num_ch_enc[row + 1] // 2 +
                    self.num_ch_dec[row]*2*(col-1), self.num_ch_dec[row] * 2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode)
            else:
                if col == 1:
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvModule(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row], self.num_ch_dec[row + 1], 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode)
                else:
                    self.convs["X_"+index+"_downsample"] = ConvModule(num_ch_enc[row+1] // 2 + self.num_ch_enc[row] + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2, 1, 1, 0, norm_cfg=None, act_cfg=None, bias=False)
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvModule(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1], 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode=padding_mode)

        if self.mobile_encoder:
            self.convs["dispConvScale0"] = ConvModule(4, self.num_output_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=None, padding_mode=padding_mode)
            self.convs["dispConvScale1"] = ConvModule(8, self.num_output_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=None, padding_mode=padding_mode)
            self.convs["dispConvScale2"] = ConvModule(24, self.num_output_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=None, padding_mode=padding_mode)
            self.convs["dispConvScale3"] = ConvModule(40, self.num_output_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=None, padding_mode=padding_mode)
        else:
            for i in range(4):
                self.convs["dispConvScale{}".format(i)] = ConvModule(self.num_ch_dec[i], self.num_output_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=None, padding_mode=padding_mode)

        #self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [F.interpolate(conv_0(high_feature), scale_factor=2, mode="nearest")]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        features = {}
        if isinstance(input_features, dict):
            input_features = list(input_features.values())

        if self.use_input_attention:
            for i in range(5):
                features[f'X_{i}0'] = self.input_attentions[5-i-1](input_features[i])
        else:
            for i in range(5):
                features[f'X_{i}0'] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](F.interpolate(x, scale_factor=2, mode="nearest"))
        # outputs[("disparity", "Scale0")] = self.sigmoid(self.convs["dispConvScale0"](x))
        # outputs[("disparity", "Scale1")] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        # outputs[("disparity", "Scale2")] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        # outputs[("disparity", "Scale3")] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))
        
        outputs[("disparity", "Scale3")] = self.convs["dispConvScale3"](features["X_22"])
        outputs[("disparity", "Scale2")] = self.convs["dispConvScale2"](features["X_13"])
        outputs[("disparity", "Scale1")] = self.convs["dispConvScale1"](features["X_04"])
        outputs[("disparity", "Scale0")] = self.convs["dispConvScale0"](x)
        return outputs