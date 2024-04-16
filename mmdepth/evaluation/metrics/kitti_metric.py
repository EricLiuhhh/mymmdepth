from typing import Any, Optional, Sequence
from collections import OrderedDict
from prettytable import PrettyTable
import numpy as np
import torch
from mmengine.logging import MMLogger, print_log
from mmengine.evaluator import BaseMetric
from mmdepth.registry import METRICS

@METRICS.register_module()
class KittiMetric(BaseMetric):
    def __init__(self, 
                 min_depth = 0.001,
                 max_depth = 80,
                 eval_crop = None,
                 depth_clamp = None,
                 scale_recovery = False,
                 collect_device: str = 'cpu', 
                 prefix: Optional[str] = None, 
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.t_valid = 0.0001
        #self.t_valid = 0.1
        self.max_depth = max_depth
        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
        ]
        self.eval_crop = eval_crop
        self.scale_recovery = scale_recovery
        self.depth_clamp = depth_clamp
        self.cnt = 0

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_depth']['data'].squeeze()
            gt = data_sample['gt_depth']['data'].squeeze()
            self.results.append(self.single_evaluate(pred, gt))

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        r = torch.concat(results).mean(axis=0)
        ret = OrderedDict()
        for name, val in zip(self.metric_name, r):
            ret[name] = val.item()
        table_data = PrettyTable()
        for key, val in ret.items():
            table_data.add_column(key, [val])
        print_log('\n' + table_data.get_string(), logger=logger)
        return ret

    # def single_evaluate(self, pred, gt):
    #     with torch.no_grad():
    #         pred_inv = 1.0 / (pred + 1e-8)
    #         gt_inv = 1.0 / (gt + 1e-8)

    #         # For numerical stability
    #         mask = gt > self.t_valid
    #         num_valid = mask.sum()

    #         pred = pred[mask]
    #         gt = gt[mask]

    #         pred_inv = pred_inv[mask]
    #         gt_inv = gt_inv[mask]

    #         pred_inv[pred <= self.t_valid] = 0.0
    #         gt_inv[gt <= self.t_valid] = 0.0

    #         # RMSE / MAE
    #         diff = pred - gt
    #         diff_abs = torch.abs(diff)
    #         diff_sqr = torch.pow(diff, 2)

    #         rmse = diff_sqr.sum() / (num_valid + 1e-8)
    #         rmse = torch.sqrt(rmse)

    #         mae = diff_abs.sum() / (num_valid + 1e-8)

    #         # iRMSE / iMAE
    #         diff_inv = pred_inv - gt_inv
    #         diff_inv_abs = torch.abs(diff_inv)
    #         diff_inv_sqr = torch.pow(diff_inv, 2)

    #         irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
    #         irmse = torch.sqrt(irmse)

    #         imae = diff_inv_abs.sum() / (num_valid + 1e-8)

    #         # Rel
    #         rel = diff_abs / (gt + 1e-8)
    #         rel = rel.sum() / (num_valid + 1e-8)

    #         # delta
    #         r1 = gt / (pred + 1e-8)
    #         r2 = pred / (gt + 1e-8)
    #         ratio = torch.max(r1, r2)

    #         del_1 = (ratio < 1.25).type_as(ratio)
    #         del_2 = (ratio < 1.25**2).type_as(ratio)
    #         del_3 = (ratio < 1.25**3).type_as(ratio)

    #         del_1 = del_1.sum() / (num_valid + 1e-8)
    #         del_2 = del_2.sum() / (num_valid + 1e-8)
    #         del_3 = del_3.sum() / (num_valid + 1e-8)

    #         result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
    #         result = torch.stack(result)
    #         result = torch.unsqueeze(result, dim=0).detach()

    #     return result
    
    def single_evaluate(self, pred, gt):
        with torch.no_grad(): 

            # For numerical stability
            mask = (gt > self.t_valid) & (gt < self.max_depth)

            if self.eval_crop is not None:
                crop_mask = torch.zeros_like(mask)
                if self.eval_crop[0] > 1:
                    crop_mask[self.eval_crop[0]:self.eval_crop[1], self.eval_crop[2]:self.eval_crop[3]] = 1
                else:
                    gt_height, gt_width = gt.shape[-2:]
                    _crop = np.array([self.eval_crop[0]*gt_height, self.eval_crop[1]*gt_height,
                             self.eval_crop[2]*gt_width, self.eval_crop[3]*gt_width]).astype(np.int32)
                    crop_mask[_crop[0]:_crop[1], _crop[2]:_crop[3]] = 1

                    # vis garg crop
                    debug = False
                    if debug:
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.imshow(gt.cpu().numpy())
                        ax = plt.gca()
                        crop_h = self.eval_crop[1]*gt_height - self.eval_crop[0]*gt_height + 1
                        crop_w = self.eval_crop[3]*gt_width - self.eval_crop[2]*gt_width + 1
                        ax.add_patch(plt.Rectangle((self.eval_crop[2]*gt_width, self.eval_crop[0]*gt_height), crop_w, crop_h, color="red", fill=False, linewidth=2))
                        plt.savefig(f'vis_garg_crop/{self.cnt}.png')
                        plt.clf()
                        self.cnt += 1

                mask = mask * crop_mask

            num_valid = mask.sum()
            pred = pred[mask]
            gt = gt[mask]
            if self.scale_recovery:
                pred *= torch.median(gt) / torch.median(pred)
            if self.depth_clamp is not None:
                pred = torch.clamp(pred, min=self.depth_clamp[0], max=self.depth_clamp[1])
                
            #pred = pred[mask]*1e3
            #gt = gt[mask]*1e3
            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)
            #pred_inv = pred_inv[mask]
            #gt_inv = gt_inv[mask]

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25**2).type_as(ratio)
            del_3 = (ratio < 1.25**3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result

