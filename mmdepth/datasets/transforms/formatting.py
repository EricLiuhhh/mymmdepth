from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
from PIL import Image
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms import BaseTransform
from mmdepth.registry import TRANSFORMS
from mmdepth.structures.depth_data_sample import DepthDataSample, PixelData

@TRANSFORMS.register_module()
class PackDepthInputs(BaseTransform):
    def __init__(self,
                 meta_keys=('img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction'),
                 input_keys=('img', 'sparse_depth', 'pos_embd', 'K', 'extrinsic', 'filtered_depth', 'prefill_depth', 'points')):
        self.meta_keys = meta_keys
        self.input_keys = input_keys

    def _img2tensor(self, img):
        if isinstance(img, list):
            # process multiple imgs in single frame
            img = np.stack(img, axis=0)
            if len(img.shape) < 4:
                img = np.expand_dims(img, -1)
            if img.flags.c_contiguous:
                img = to_tensor(img).permute(0, 3, 1, 2).contiguous()
            else:
                img = to_tensor(
                    np.ascontiguousarray(img.transpose(0, 3, 1, 2)))
        else:
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # `torch.permute()` rather than `np.transpose()`.
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if img.flags.c_contiguous:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            else:
                img = to_tensor(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))
        return img
    
    def transform(self, results: Dict) -> Dict:
        packed_results = dict(inputs={})
        for k in self.input_keys:
            if k == 'img':
                all_matches = [s for s in results.keys() if (("img" in s) and isinstance(results[s], (np.ndarray, torch.Tensor, list)))]
                for match in all_matches:
                    packed_results['inputs'][match] = self._img2tensor(results[match])
            elif k in results:
                if 'points' in k:
                    packed_results['inputs'][k] = results[k].tensor
                else:
                    packed_results['inputs'][k] = self._img2tensor(results[k])

        data_sample = DepthDataSample()
        if 'gt_depth' in results:
            data_sample.gt_depth = PixelData(**dict(data=self._img2tensor(results['gt_depth'])))
        img_meta = {}
        for key in self.meta_keys:
            if key not in results:
                continue
            # assert key in results, f'`{key}` is not found in `results`, ' \
            #     f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample
        return packed_results