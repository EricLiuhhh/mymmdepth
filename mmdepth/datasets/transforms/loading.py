from typing import Dict, List, Optional, Tuple, Union
import os
import copy
import numpy as np
import h5py
from PIL import Image
import skimage
import torch
from mmcv.transforms import BaseTransform, LoadImageFromFile, Normalize
from mmdepth.registry import TRANSFORMS
from mmdepth.models.layers.positional_encoding import PositionalEncoding
from mmdepth.structures import ImageList
from .utils import generate_depth_map_kitti, generate_depth_map_vod, read_calib_file_vod

@TRANSFORMS.register_module()
class LoadDepth(BaseTransform):
    '''
    Input Keys:
        depth_path
        depth_gt_path
    Output Keys:
        sparse_depth
        gt_depth
    '''
    def __init__(self) -> None:
        super().__init__()

    def _read_depth(self, file_name):
        # loads depth map D from 16 bits png file as a numpy array,
        # refer to readme file in KITTI dataset
        assert os.path.exists(file_name), "file not found: {}".format(file_name)
        image_depth = np.array(Image.open(file_name))

        # Consider empty depth
        assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
            "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

        image_depth = image_depth.astype(np.float32) / 256.0
        return image_depth
    
    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        if 'depth_path' in results:
            results['sparse_depth'] = self._read_depth(results['depth_path'])
        if 'depth_gt_path' in results:
            results['gt_depth'] = self._read_depth(results['depth_gt_path'])
        return results
    
@TRANSFORMS.register_module()
class LoadTripletImages(LoadImageFromFile):
    '''
    Input Keys:
        img_path
    Output Keys:
        img
    Note: file names need to be contiguous
    '''
    def __init__(self,
                 backend_args) -> None:
        super().__init__(backend_args=backend_args)

    def _load_neighbor_images(self, file_path:str):
        # root/seq/image_02(3)/data/file_idx.png
        tokens = file_path.split('/')
        file_idx, suffix = tokens[-1].split('.')
        prev_file = os.path.join(*tokens[:-1], f'{int(file_idx)-1:0{len(file_idx)}d}.{suffix}')
        next_file = os.path.join(*tokens[:-1], f'{int(file_idx)+1:0{len(file_idx)}d}.{suffix}')
        if os.path.exists(prev_file) and os.path.exists(next_file):
            prev_img = super().transform(dict(img_path=prev_file))['img']
            next_img = super().transform(dict(img_path=next_file))['img']
            return prev_img, next_img
        return None, None
    
    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        prev_img, next_img = self._load_neighbor_images(results['img_path'])
        if prev_img is None:
            return None
        else:
            cur_img = super().transform(results)['img']
            results['img'] = ImageList([prev_img, cur_img, next_img])
        return results

@TRANSFORMS.register_module()
class LoadCalibKitti(BaseTransform):
    '''
    Input Keys:
        calib_path
    Output Keys:
        K
    '''
    def __init__(self, K=None) -> None:
        super().__init__()
        if isinstance(K, list):
            K = np.array(K).astype(np.float32)
        elif isinstance(K, np.ndarray):
            pass
        elif isinstance(K, torch.Tensor):
            K = K.numpy()
        if K is not None:
            assert K.shape == (3, 3)
        self.K = K

    # for PENet, with bug
    # def _read_calib(self, file_name):
    #     #with open(file_name, 'r') as f:
    #     with open('/home/lhs/depth_est_comp/PENet_ICRA2021/dataloaders/calib_cam_to_cam.txt', 'r') as f:
    #         lines = f.readlines()
    #     P_rect_line = lines[25]

    #     Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    #     Proj = np.reshape(np.array([float(p) for p in Proj_str]),
    #                     (3, 4)).astype(np.float32)
    #     K = Proj[:3, :3]  # camera matrix

    #     # note: we will take the center crop of the images during augmentation
    #     # that changes the optical centers, but not focal lengths
    #     K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    #     K[1, 2] = K[1, 2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    #     return K

    def _read_calib(self, file_name):
        # train
        if os.path.basename(file_name) == 'calib_cam_to_cam.txt':
            with open(file_name, 'r') as f:
                lines = f.readlines()
            P_rect_line = lines[25]

            Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
            Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                            (3, 4)).astype(np.float32)
            K = Proj[:3, :3]  # camera matrix
        # val / test
        else:
            with open(file_name, 'r') as f:
                line = f.readline().strip()
            P_rect_line = line.split(' ')
            Proj = np.reshape(np.array([float(p) for p in P_rect_line]),
                            (3, 3)).astype(np.float32)
            K = Proj[:3, :3]
        return K
    
    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        if self.K is not None:
            results['K'] = copy.deepcopy(self.K)
        elif 'calib_path' in results:
            results['K'] = self._read_calib(results['calib_path'])
        return results

@TRANSFORMS.register_module()
class LoadCalibVoD(BaseTransform):
    '''
    Input Keys:
        calib_path
    Output Keys:
        K
    '''
    def __init__(self, K=None, extrinsic=None) -> None:
        super().__init__()
        if isinstance(K, list):
            K = np.array(K).astype(np.float32)
            assert K.shape==(3, 3)
        self.K = K

        if isinstance(extrinsic, list):
            extrinsic = np.array(extrinsic).astype(np.float32)
            assert extrinsic.shape==(4, 4)
        self.extrinsic = extrinsic


    def _read_calib(self, calib_path):
        K, extrinsic = read_calib_file_vod(calib_path)
        return K[:3, :3], extrinsic
    
    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        if (self.K is not None) and (self.extrinsic is not None):
            results['K'], results['extrinsic'] = copy.deepcopy(self.K), copy.deepcopy(self.extrinsic)
        else:
            results['K'], results['extrinsic'] = self._read_calib(results['calib_path'])
            if self.K is not None:
                results['K'] = copy.deepcopy(self.K)
            if self.extrinsic is not None:
                results['extrinsic'] = copy.deepcopy(self.extrinsic)
        return results


@TRANSFORMS.register_module()
class GenPosEmbd(BaseTransform):
    '''
    Input Keys:
        None
    Output Keys:
        pos_embd
    '''
    def __init__(self, height, width, scale, offset) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.pos_gen = PositionalEncoding(scale=scale, offset=offset)
        #self.pos_gen = AddCoordsNp(height, width)
    
    def transform(self, results: Dict) -> Union[Dict, Tuple[List, List], None]:
        mask = torch.zeros((1, self.height, self.width), dtype=torch.bool)
        results['pos_embd'] = self.pos_gen(mask).squeeze()
        #results['pos_embd'] = self.pos_gen.call()
        return results
    
class AddCoordsNp():
	"""Add coords to a tensor"""
	def __init__(self, x_dim=64, y_dim=64, with_r=False):
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.with_r = with_r

	def call(self):
		"""
		input_tensor: (batch, x_dim, y_dim, c)
		"""
		#batch_size_tensor = np.shape(input_tensor)[0]

		xx_ones = np.ones([self.x_dim], dtype=np.int32)
		xx_ones = np.expand_dims(xx_ones, 1)

		#print(xx_ones.shape)

		xx_range = np.expand_dims(np.arange(self.y_dim), 0)
		#xx_range = np.expand_dims(xx_range, 1)

		#print(xx_range.shape)

		xx_channel = np.matmul(xx_ones, xx_range)
		xx_channel = np.expand_dims(xx_channel, -1)

		yy_ones = np.ones([self.y_dim], dtype=np.int32)
		yy_ones = np.expand_dims(yy_ones, 0)

		#print(yy_ones.shape)

		yy_range = np.expand_dims(np.arange(self.x_dim), 1)
		#yy_range = np.expand_dims(yy_range, -1)

		#print(yy_range.shape)

		yy_channel = np.matmul(yy_range, yy_ones)
		yy_channel = np.expand_dims(yy_channel, -1)

		xx_channel = xx_channel.astype('float32') / (self.y_dim - 1)
		yy_channel = yy_channel.astype('float32') / (self.x_dim - 1)

		xx_channel = xx_channel*2 - 1
		yy_channel = yy_channel*2 - 1
	

		#xx_channel = xx_channel.repeat(batch_size_tensor, axis=0)
		#yy_channel = yy_channel.repeat(batch_size_tensor, axis=0)

		ret = np.concatenate([xx_channel, yy_channel], axis=-1)

		if self.with_r:
			rr = np.sqrt( np.square(xx_channel-0.5) + np.square(yy_channel-0.5))
			ret = np.concatenate([ret, rr], axis=-1)

		return ret
     
@TRANSFORMS.register_module()
class GenGTDepthKitti(BaseTransform):
    def __init__(self, shape, keep_velo_depth=False) -> None:
        super().__init__()
        self.shape = shape
        self.keep_velo_depth = keep_velo_depth

    def transform(self, results: Dict):
        side_map = {'l': 2, 'r': 3}
        gt_depth = generate_depth_map_kitti(results['calib_path'], results['calib_velo_path'], results['velodyne_path'], side_map[results['side']], self.keep_velo_depth)
        gt_depth = skimage.transform.resize(
            gt_depth, self.shape[::-1], order=0, preserve_range=True, mode='constant')
        results['gt_depth'] = gt_depth
        return results
    
@TRANSFORMS.register_module()
class GenGTDepthVod(BaseTransform):
    def __init__(self, shape, keep_velo_depth=False) -> None:
        super().__init__()
        self.shape = shape
        self.keep_velo_depth = keep_velo_depth

    def transform(self, results: Dict):
        side_map = {'l': 2, 'r': 3}
        gt_depth = generate_depth_map_vod(results['calib_path'], results['velodyne_path'], side_map[results['side']], self.keep_velo_depth)
        gt_depth = skimage.transform.resize(
            gt_depth, self.shape[::-1], order=0, preserve_range=True, mode='constant')
        results['gt_depth'] = gt_depth
        return results

@TRANSFORMS.register_module()
class LoadH5(BaseTransform):
    def __init__(self, key_map=None, decompress_depth=True) -> None:
        super().__init__()
        self.key_map = key_map
        self.decompress_depth = decompress_depth

    def transform(self, results: Dict):
        data_path = results['h5_path']
        with h5py.File(data_path, 'r') as f:
            if self.key_map is not None:
                for n in f:
                    if n in self.key_map:
                        results[self.key_map[n]] = np.array(f[n])
            else:
                for n in f:
                    results[n] = np.array(f[n])
        
        for k in results:
            if 'depth' in k:
                results[k] = results[k].astype(np.float32) / 256.0
        return results