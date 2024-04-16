from typing import List, Tuple
import copy
import cv2
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
from mmdepth.utils.typing_utils import OptConfigType, OptMultiConfig, ConfigType, OptSampleList, SampleList
from .base_estimator import BaseEstimator
from ..utils.misc import disp_to_depth

@MODELS.register_module()
class Monodepth2(BaseEstimator):
    def __init__(self, 
                 batch_size,
                 img_encoder: ConfigType,
                 img_decoder: ConfigType,
                 img_decode_guide: ConfigType,
                 pose_encoder: ConfigType = None,
                 pose_decoder: ConfigType = None,
                 data_preprocessor: OptConfigType = None, 
                 conv_out_locations = ('g1', 'g2', 'g3', 'l5'),
                 input_shape = (1024, 320),
                 loss_cfg = dict(),
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        self.img_encoder = MODELS.build(img_encoder)
        self.img_decoder = MODELS.build(img_decoder)
        self.img_decode_guide_map = img_decode_guide['guide_map']
        self.guide_map[('img_encoder', list(zip(*self.img_decode_guide_map))[0])] = ('img_decoder', list(zip(*self.img_decode_guide_map))[1])
        self.build_guide(img_decode_guide, self.img_encoder, self.img_decoder)
        self.img_decoder.build_conv_outs(conv_out_locations)
        if self.training:
            assert pose_decoder is not None
            self.pose_encoder = MODELS.build(pose_encoder)
            self.pose_decoder = MODELS.build(pose_decoder)
        
            w, h = input_shape
            backproj_cfg = dict(type='BackprojectDepth', batch_size=batch_size, height=h, width=w)
            proj3d_cfg = dict(type='Project3D', batch_size=batch_size, height=h, width=w)
            self.backproject_depth = MODELS.build(backproj_cfg)
            self.project_3d = MODELS.build(proj3d_cfg)
            self.ssim_loss = MODELS.build(dict(type='SSIMLoss'))
            self.smoothness_loss = MODELS.build(dict(type='EdgeAwareSmoothnessLoss'))
            self.smoothness_weight = loss_cfg.get('smoothness_weight', 1e-3)

    def init_weights(self):
        super().init_weights()
        ms = [self.img_decoder, self.pose_decoder]
        for m in ms:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                        if fan_in != 0:
                            bound = 1 / math.sqrt(fan_in)
                            init.uniform_(module.bias, -bound, bound)

    def compute_reprojection_loss(self, pred, target, use_ssim=True):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if not use_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim_loss(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def loss(self, batch_inputs, batch_data_samples):
        disp_preds, Ts = self._forward(batch_inputs, batch_data_samples)
        img1, img2, img3 = batch_inputs['img_pyr_1'][:, 0, ...], batch_inputs['img_pyr_1'][:, 1, ...], batch_inputs['img_pyr_1'][:, 2, ...]
        
        debug = False
        if debug:
            import numpy as np
            import matplotlib.pyplot as plt
            plt.figure()
            for i in range(3):
                for j in range(4):
                    plt.subplot(3, 4, i*4+j+1)
                    s = 2**j
                    img_pyr = batch_inputs[f'img_pyr_{s}'][0, i, ...]
                    img_pyr = img_pyr.permute(1, 2, 0).cpu().numpy()
                    plt.imshow(img_pyr)
            plt.savefig('z_img_pyr.png')

            plt.figure()
            for i in range(3):
                plt.subplot(1, 3, i+1)
                img = batch_inputs['img'][0, i, ...]
                img = img.permute(1, 2, 0).cpu().numpy()
                plt.imshow(img)
            plt.savefig('z_img.png')

        Is = [img1, img3]
        losses = {}
        num_scales = len(disp_preds)
        for i, disp in enumerate(disp_preds):
            disp = torch.sigmoid(disp)
            scale = 2**(num_scales-i-1)
            disp_up = F.interpolate(
            disp, size=disp_preds[-1].shape[-2:], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp_up, 0.1, 100)
            Is_reproj = []
            reproj_losses = []
            identity_reproj_losses = []
            for j in range(2):
                T = Ts[j]
                K = batch_inputs['K'].squeeze(1)
                cam_points = self.backproject_depth(depth, torch.linalg.pinv(K))
                pix_coords = self.project_3d(cam_points, K, T)
                Is_reproj.append(F.grid_sample(
                    Is[j],
                    pix_coords,
                    padding_mode="border"))
            target = img2
            for j in range(2):
                pred = Is_reproj[j]
                reproj_losses.append(self.compute_reprojection_loss(pred, target))
            for j in range(2):
                identity_reproj_losses.append(self.compute_reprojection_loss(Is[j], target))
            reproj_loss = torch.cat(reproj_losses, dim=1)
            identity_reproj_loss = torch.cat(identity_reproj_losses, dim=1)
            identity_reproj_loss += torch.randn(identity_reproj_loss.shape, device=identity_reproj_loss.device) * 0.00001
            combined = torch.cat([identity_reproj_loss, reproj_loss], dim=1)
            to_optimise, idxs = torch.min(combined, dim=1)
            #moving_mask = (idxs > identity_reproj_loss.shape[1]-1).float()
            min_reproj_loss = to_optimise.mean()
            norm_disp = disp / (disp.mean(2, True).mean(3, True) + 1e-7)
            smooth_loss = self.smoothness_loss(norm_disp, batch_inputs[f'img_pyr_{scale}'][:, 1, ...])
            smooth_loss = self.smoothness_weight * smooth_loss / (scale)
            losses[f'reproj_loss_s{scale}'] = min_reproj_loss / num_scales
            losses[f'smooth_loss_s{scale}'] = smooth_loss / num_scales
        return losses

    def _norm_img(self, img: torch.Tensor):
        return (img - 0.45) / 0.225
    
    def _forward(self, batch_inputs, batch_data_samples=None):
        # img shape: B * N * C * H * W
        pose1, pose2 = None, None
        if len(batch_inputs['img'].shape) == 5:
            img = batch_inputs['img'][:, 1, ...]
            if hasattr(self, 'pose_encoder'):
                img1, img2, img3 = batch_inputs['img'][:, 0, ...], batch_inputs['img'][:, 1, ...], batch_inputs['img'][:, 2, ...]
                pair1 = torch.cat([img1, img2], dim=1)
                pair2 = torch.cat([img2, img3], dim=1)
                pair1 = self._norm_img(pair1)
                pair2 = self._norm_img(pair2)
                pose1 = self.pose_decoder(self.pose_encoder(dict(feats=pair1))[f'l{len(self.pose_encoder.layers)-1}'], invert=True)
                pose2 = self.pose_decoder(self.pose_encoder(dict(feats=pair2))[f'l{len(self.pose_encoder.layers)-1}'])
        else:
            img = batch_inputs['img']
        img_encoder_inputs = dict(feats=self._norm_img(img))
        img_encoder_outputs = self.img_encoder(img_encoder_inputs)

        img_decoder_inputs = dict(feats=img_encoder_outputs[f'l{len(self.img_encoder.layers)-1}'])
        for i, k in enumerate(self.img_decode_guide_map):
            img_decoder_inputs[f'guide{i}'] = img_encoder_outputs[f'{k[0]}']
        img_decoder_outputs = self.img_decoder(img_decoder_inputs)
        return list(img_decoder_outputs.values()), (pose1, pose2)
    
    def predict(self, batch_inputs, batch_data_samples=None) -> SampleList:
        disp_preds, _ = self._forward(batch_inputs, batch_data_samples)
        disp_pred = disp_preds[-1]
        disp_pred, depth_pred = disp_to_depth(torch.sigmoid(disp_pred), 0.1, 100)
        # depth_pred = torch.clamp(F.interpolate(
        #     depth_pred, batch_data_samples[0].gt_depth.shape[-2:], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = 1.0 / F.interpolate(
            disp_pred, size=batch_data_samples[0].gt_depth.shape[-2:], mode="bilinear", align_corners=False)

        for i in range(len(batch_data_samples)):
            batch_data_samples[i].pred_depth = PixelData(**dict(data=depth_pred[i]))
        return batch_data_samples