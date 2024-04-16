import os
import numbers
from numbers import Number
import math
import copy
from typing import Dict, Iterable, List, Optional, Tuple, Union, Sequence, Union
from functools import partial
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms as TF
import mmcv
from mmcv.transforms import BaseTransform, CenterCrop, RandomResize, RandomFlip, Normalize
from mmcv.transforms.utils import cache_randomness
from mmdepth.registry import TRANSFORMS
from mmdepth.models.layers.positional_encoding import PositionalEncoding
from .utils import apply_covariants, outlier_removal, fill_in_fast

@TRANSFORMS.register_module()
class BottomCrop(BaseTransform):
    def __init__(self, crop_size: Union[int, Tuple[int, int]], covariants: Optional[Dict] = None) -> None:
        super().__init__()
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        if covariants is None:
            self.covariants = {'img': {}, 'sparse_depth': {}, 'gt_depth': {}, 'pos_embd': {}}
        else:
            self.covariants = covariants
            assert 'crop_size' not in covariants['img'], 'The covariation of `crop_size` is not supported.'

    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        img = results['img']
        # img.shape has length 2 for grayscale, length 3 for color
        height, width = img.shape[:2]
        crop_height, crop_width = self.crop_size[1], self.crop_size[0]
        if crop_width == 'auto':
            crop_width = width
        assert (crop_height <= height) and (crop_width <= width), 'Crop size should be smaller than image size.'
        h_init = height - crop_height
        w_init = (width - crop_width) // 2
        bboxes = np.array([w_init, h_init, w_init+crop_width-1, h_init+crop_height-1])

        if 'K' in results:
            if 'K_ori' not in results:
                # back up
                results['K_ori'] = copy.deepcopy(results['K'])
            results['K'][..., 1, 2] -= h_init
            results['K'][..., 0, 2] -= w_init

        # crop
        apply_func = partial(mmcv.imcrop, bboxes=bboxes)
        results = apply_covariants(results, apply_func, self.covariants)
        img_shape = results['img'].shape[:2]  # type: ignore
        results['img_shape'] = img_shape
        results['pad_shape'] = img_shape
        return results

@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast and saturation of an image.

    Modified from
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
    Licensed under the BSD 3-Clause License.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        brightness (float | Sequence[float] (min, max)): How much to jitter
            brightness. brightness_factor is chosen uniformly from
            ``[max(0, 1 - brightness), 1 + brightness]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        contrast (float | Sequence[float] (min, max)): How much to jitter
            contrast. contrast_factor is chosen uniformly from
            ``[max(0, 1 - contrast), 1 + contrast]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        saturation (float | Sequence[float] (min, max)): How much to jitter
            saturation. saturation_factor is chosen uniformly from
            ``[max(0, 1 - saturation), 1 + saturation]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        hue (float | Sequence[float] (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from ``[-hue, hue]`` (0 <= hue
            <= 0.5) or the given ``[min, max]`` (-0.5 <= min <= max <= 0.5).
            Defaults to 0.
        backend (str): The backend to operate the image. Defaults to 'pillow'
    """

    def __init__(self,
                 brightness: Union[float, Sequence[float]] = 0.,
                 contrast: Union[float, Sequence[float]] = 0.,
                 saturation: Union[float, Sequence[float]] = 0.,
                 hue: Union[float, Sequence[float]] = 0.,
                 backend='pillow'):
        self.brightness = self._set_range(brightness, 'brightness')
        self.contrast = self._set_range(contrast, 'contrast')
        self.saturation = self._set_range(saturation, 'saturation')
        self.hue = self._set_range(hue, 'hue', center=0, bound=(-0.5, 0.5))
        self.backend = backend

    def _set_range(self, value, name, center=1, bound=(0, float('inf'))):
        """Set the range of magnitudes."""
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = (center - float(value), center + float(value))

        if isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                value = np.clip(value, bound[0], bound[1])
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                logger.warning(f'ColorJitter {name} values exceed the bound '
                               f'{bound}, clipped to the bound.')
        else:
            raise TypeError(f'{name} should be a single number '
                            'or a list/tuple with length 2.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        else:
            value = tuple(value)

        return value

    @cache_randomness
    def _rand_params(self):
        """Get random parameters including magnitudes and indices of
        transforms."""
        trans_inds = np.random.permutation(4)
        b, c, s, h = (None, ) * 4

        if self.brightness is not None:
            b = np.random.uniform(self.brightness[0], self.brightness[1])
        if self.contrast is not None:
            c = np.random.uniform(self.contrast[0], self.contrast[1])
        if self.saturation is not None:
            s = np.random.uniform(self.saturation[0], self.saturation[1])
        if self.hue is not None:
            h = np.random.uniform(self.hue[0], self.hue[1])

        return trans_inds, b, c, s, h
    
    def _apply_jitter(self, img, trans_inds, brightness, contrast, saturation, hue):
        for index in trans_inds:
            if index == 0 and brightness is not None:
                img = mmcv.adjust_brightness(
                    img, brightness, backend=self.backend)
            elif index == 1 and contrast is not None:
                img = mmcv.adjust_contrast(img, contrast, backend=self.backend)
            elif index == 2 and saturation is not None:
                img = mmcv.adjust_color(
                    img, alpha=saturation, backend=self.backend)
            elif index == 3 and hue is not None:
                img = mmcv.adjust_hue(img, hue, backend=self.backend)
        return img
    
    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: ColorJitter results, 'img' key is updated in result dict.
        """
        img = results['img']
        trans_inds, brightness, contrast, saturation, hue = self._rand_params()
        if isinstance(img, (list, tuple)):
            for i in range(len(img)):
                results['img'][i] = self._apply_jitter(img[i], trans_inds, brightness, contrast, saturation, hue)
        else:
            results['img'] = self._apply_jitter(img, trans_inds, brightness, contrast, saturation, hue)
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation}, '
        repr_str += f'hue={self.hue})'
        return repr_str

@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Random Rotate images.

    Args:
        angle (float, optional): The angle used for rotate. Positive values
            stand for clockwise rotation. If None, generate from
            ``magnitude_range``, see :class:`BaseAugTransform`.
            Defaults to None.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If None, the center of the image will be used.
            Defaults to None.
        scale (float): Isotropic scale factor. Defaults to 1.0.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 0.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
        covariants (dict): support 'interpolation'
    """

    def __init__(self,
                 angle: Optional[float] = None,
                 center: Optional[Tuple[float]] = None,
                 scale: float = 1.0,
                 pad_val: Union[int, Sequence[int]] = 0,
                 interpolation: str = 'bilinear',
                 covariants: Dict = None):
        super().__init__()
        assert (angle is not None), \
            'Please specify `angle`.'

        self.angle = angle
        self.center = center
        self.scale = scale
        if isinstance(pad_val, Sequence):
            self.pad_val = tuple(pad_val)
        else:
            self.pad_val = pad_val

        self.interpolation = interpolation
        if covariants is None:
            self.covariants = {'img': {'interpolation':self.interpolation}, 'sparse_depth': {'interpolation':'nearest'}, 'gt_depth': {'interpolation':'nearest'}}
        else:
            self.covariants = covariants

    @staticmethod
    def _apply_rotate(img, angle, center, scale, border_value, interpolation):
        img_rotated = mmcv.imrotate(
            img,
            angle,
            center=center,
            scale=scale,
            border_value=border_value,
            interpolation=interpolation)
        img_rotated = img_rotated.astype(img.dtype)
        return img_rotated

    def transform(self, results):
        """Apply transform to results."""
        angle = np.random.uniform(-self.angle, self.angle)
        apply_func = partial(self._apply_rotate, angle=angle, center=self.center, scale=self.scale, border_value=self.pad_val)
        results = apply_covariants(results, apply_func, self.covariants)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angle={self.angle}, '
        repr_str += f'center={self.center}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation}{self.extra_repr()})'
        return repr_str

@TRANSFORMS.register_module()
class RandomFlipBundle(RandomFlip):
    def __init__(self, prob: Optional[Union[float, Iterable[float]]] = None, direction: Union[str, Sequence[Optional[str]]] = 'horizontal', swap_seg_labels: Optional[Sequence] = None, covariants: Optional[dict] = None) -> None:
        super().__init__(prob, direction, swap_seg_labels)
        if covariants is None:
            self.covariants = {'img': {}, 'sparse_depth': {}, 'gt_depth': {}}
        else:
            self.covariants = covariants

    @staticmethod
    def _apply_flip(img, direction):
        return mmcv.imflip(
            img, direction=direction)

    def transform(self, results: dict) -> dict:
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            if 'K' in results:
                if 'K_ori' not in results:
                    # back up
                    results['K_ori'] = copy.deepcopy(results['K'])
                results['K'][..., 0, 2] = results['img'].shape[1] - results['K'][..., 0, 2]

            apply_func = partial(self._apply_flip, direction=results['flip_direction'])
            results = apply_covariants(results, apply_func, self.covariants)
        return results

@TRANSFORMS.register_module()
class RandomScaleBundle(RandomResize):
    def __init__(self,         
                 scale: Tuple[int, int],
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2',
                 scale_depth = True,
                 covariants: dict = None) -> None:
        self.scale = scale
        self.backend = backend
        self.interpolation = interpolation
        if covariants is None:
            self.covariants = {'img': {'interpolation':self.interpolation}, 'sparse_depth': {'interpolation':'nearest'}, 'gt_depth': {'interpolation':'nearest'}}
        else:
            self.covariants = covariants
        self.scale_depth = scale_depth

    @staticmethod
    def _apple_resize(img, scale, interpolation, backend):
        img_rescaled, _ = mmcv.imrescale(
            img,
            scale,
            interpolation=interpolation,
            return_scale=True,
            backend=backend)
        return img_rescaled

    def transform(self, results: dict) -> dict:
        results['scale'] = np.random.uniform(self.scale[0], self.scale[1])
        h_ori, w_ori = results['img'].shape[:2]
        apply_func = partial(self._apple_resize, scale=results['scale'], backend=self.backend)
        results = apply_covariants(results, apply_func, self.covariants)
        results['img_shape'] = results['img'].shape[:2]
        h_rescaled, w_rescaled = results['img_shape']
        results['scale_factor'] = (w_rescaled/w_ori, h_rescaled/h_ori)
        # rescale intrinsic
        if 'K' in results:
            if 'K_ori' not in results:
                # back up
                results['K_ori'] = copy.deepcopy(results['K'])
            results['K'][..., :2, :] *= results['scale']
        # rescale depth
        if self.scale_depth:
            for k in self.covariants.keys():
                if 'depth' in k and k in results:
                    results[k] /= results['scale']
        return results
    
@TRANSFORMS.register_module()
class ResizeBundle(RandomResize):
    def __init__(self,         
                 size: Tuple[int, int],
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2',
                 covariants: dict = None) -> None:
        self.size = size
        self.backend = backend
        self.interpolation = interpolation
        if covariants is None:
            self.covariants = {'img': {'interpolation':self.interpolation}, 'sparse_depth': {'interpolation':'nearest'}, 'gt_depth': {'interpolation':'nearest'}}
        else:
            self.covariants = covariants

    @staticmethod
    def _apple_resize(img, size, interpolation, backend):
        img_resized, _, _ = mmcv.imresize(
            img,
            size,
            interpolation=interpolation,
            return_scale=True,
            backend=backend)
        # func = TF.Resize(size, interpolation=Image.LANCZOS)
        # img_resized = np.array(func(Image.fromarray(img)))
        return img_resized

    def transform(self, results: dict) -> dict:
        h_ori, w_ori = results['img'].shape[:2]
        apply_func = partial(self._apple_resize, size=self.size, backend=self.backend)
        results = apply_covariants(results, apply_func, self.covariants)
        results['img_shape'] = results['img'].shape[:2]
        h_rescaled, w_rescaled = results['img_shape']
        results['scale_factor'] = (w_rescaled/w_ori, h_rescaled/h_ori)
        # rescale intrinsic
        if 'K' in results:
            if 'K_ori' not in results:
                # back up
                results['K_ori'] = copy.deepcopy(results['K'])
            results['K'][..., 0, :] *= results['scale_factor'][0]
            results['K'][..., 1, :] *= results['scale_factor'][1]
        return results
    
@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Crop the given Image at a random location.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        crop_size (int | Sequence): Desired output size of the crop. If
            crop_size is an int instead of sequence like (h, w), a square crop
            (crop_size, crop_size) is made.
    """

    def __init__(self,
                 crop_size: Union[Sequence, int], covariants: dict = None):
        if isinstance(crop_size, Sequence):
            assert len(crop_size) == 2
            assert crop_size[0] > 0 and crop_size[1] > 0
            self.crop_size = crop_size
        else:
            assert crop_size > 0
            self.crop_size = (crop_size, crop_size)
        if covariants is None:
            self.covariants = {'img': {}, 'sparse_depth': {}, 'gt_depth': {}, 'pos_embd': {}}
        else:
            self.covariants = covariants

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        target_w, target_h = self.crop_size
        if w == target_w and h == target_h:
            return 0, 0, h, w

        offset_h = np.random.randint(0, h - target_h + 1)
        offset_w = np.random.randint(0, w - target_w + 1)

        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        assert (self.crop_size[0]<=img.shape[1]) and (self.crop_size[1]<=img.shape[0]), 'Crop size should be smaller than image size'

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        apply_func = partial(mmcv.imcrop, bboxes=np.array([offset_w, offset_h, offset_w + target_w - 1, offset_h + target_h - 1]))
        results = apply_covariants(results, apply_func, self.covariants)
        results['img_shape'] = results['img'].shape[:2]
        if 'K' in results:
            if 'K_ori' not in results:
                # back up
                results['K_ori'] = copy.deepcopy(results['K'])
            results['K'][..., 1, 2] -= offset_h
            results['K'][..., 0, 2] -= offset_w
        return results

@TRANSFORMS.register_module()
class AddPosition(BaseTransform):
    def __init__(self,
                 normalize=True,
                 scale=2,
                 offset=-1
                 ) -> None:
        super().__init__()
        self.pe = PositionalEncoding(normalize, scale, offset)

    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        assert 'img' in results
        img = results['img']
        mask = torch.zeros((1, img.shape[0], img.shape[1]), dtype=torch.bool)
        pos = self.pe.forward(mask)[0]
        results['pos_embd'] = pos
        return results

@TRANSFORMS.register_module()
class FilterDepth(BaseTransform):
    '''
    Input Keys:
        sparse_depth
    Output Keys:
        filtered_depth
    '''
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        sparse_depth = results['sparse_depth']
        filtered_depth, _ = outlier_removal(sparse_depth)
        results['filtered_depth'] = filtered_depth
        return results

@TRANSFORMS.register_module()
class PreFillDepth(BaseTransform):
    '''
    Input Keys:
        sparse_depth
    Output Keys:
        prefill_depth
    '''
    def __init__(self,
                 max_depth=100) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.filler = fill_in_fast

    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        sparse_depth = results['sparse_depth']
        prefill_depth = self.filler(np.copy(sparse_depth), max_depth=self.max_depth, extrapolate=True, blur_type='gaussian').astype(np.float32)
        results['prefill_depth'] = prefill_depth
        return results
    
@TRANSFORMS.register_module()
class BackupImage(BaseTransform):
    def __init__(self, tag) -> None:
        super().__init__()
        self.tag = tag
    
    def transform(self, results: Dict):
        results[f'img_backup_{self.tag}'] = copy.deepcopy(results['img'])

@TRANSFORMS.register_module()
class BuildImagePyramid(BaseTransform):
    def __init__(self, num_scales=4, interpolation='bilinear') -> None:
        super().__init__()
        self.num_scales = num_scales
        self.resize = []
        for i in range(1, self.num_scales):
            s = 2 ** i
            self.resize.append(partial(mmcv.imrescale, scale=1/s, interpolation=interpolation,
                                            return_scale=False))
    def transform(self, results: Dict):
        img = results['img']
        results[f'img_pyr_1'] = copy.deepcopy(img)
        for i, func in enumerate(self.resize):
            if isinstance(img, (list, tuple)):
                results[f'img_pyr_{2**(i+1)}'] = list(map(func, img))
            else:
                results[f'img_pyr_{2**(i+1)}'] = func(img)
        return results
    
@TRANSFORMS.register_module()
class DepthRangeFilter(BaseTransform):
    def __init__(self, min_depth=0.0, max_depth=80.0, keys=['gt_depth', 'sparse_depth']) -> None:
        super().__init__()
        self.keys = keys
        self.min_depth = min_depth
        self.max_depth = max_depth

    def transform(self, results: Dict):
        for k in self.keys:
            if k in results:
                mask = (results[k] > self.max_depth) | (results[k] < self.min_depth)
                results[k][mask] = 0
        return results
    
@TRANSFORMS.register_module()
class ConvertColor(BaseTransform):
    '''
    Convert color

    Args:
        flag: rgb2bgr, bgr2rgb, rgb2gray
    
    '''
    _flag_map = {
        'rgb2bgr': cv2.COLOR_RGB2BGR,
        'bgr2rgb': cv2.COLOR_BGR2RGB,
        'rgb2gray': cv2.COLOR_RGB2GRAY    
    }
    def __init__(self, flag) -> None:
        super().__init__()
        self.flag = flag

    def transform(self, results: Dict):
        results['img'] = cv2.cvtColor(results['img'], self._flag_map[self.flag])
        return results