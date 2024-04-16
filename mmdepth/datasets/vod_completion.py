import os
import glob
import pickle
from typing import List, Union
from mmengine.dataset import BaseDataset
from mmdepth.registry import DATASETS

@DATASETS.register_module()
class VoDCompletion(BaseDataset):
    '''
    
    {
        'metainfo': {
            xxx
        },
        'data_list': [
            {
                'img_path': xxx,
                'depth_path': xxx,
                'depth_height': xxx,
                'depth_width': xxx,
                'calib_path': xxx,
            }
        ]
    }
    '''
    side_map = {'l': 2, 'r': 3}

    @classmethod
    def create_data(self, split, root_path, out_path, pkl_prefix, split_file, point_type='radar_5frames'):
        with open(split_file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        items = []
        for line in lines:
            seq = line[0]
            frame_idx = line[1]
            item = {
            # ann dir
            'depth_path': os.path.join(root_path, seq, 'proj_depth', point_type, f'{int(frame_idx):05d}.png'),
            'depth_gt_path': os.path.join(root_path, seq, 'proj_depth', 'lidar', f'{int(frame_idx):05d}.png'),
            # raw data dir
            'lidar_points': {'lidar_path': os.path.join(root_path, seq, 'lidar', f'{int(frame_idx):05d}.bin')},
            'input_points': {'lidar_path': os.path.join(root_path, seq, point_type, f'{int(frame_idx):05d}.bin')},
            # 'velodyne_path': os.path.join(root_path, seq, point_type, f'{int(frame_idx):05d}.bin'),
            'calib_path': os.path.join(root_path, seq, 'calib', point_type, f'{int(frame_idx):05d}.txt'),
            'img_path': os.path.join(root_path, seq, f'image_0{self.side_map[line[-1]]}', f'{int(frame_idx):05d}.jpg'),
            'side': line[-1]
            }
            items.append(item)
        infos = {
            'metainfo': {},
            'data_list': items
        }
        with open(os.path.join(out_path, f'{pkl_prefix}_{point_type}_infos_{split}.pkl'), 'wb') as f:
            pickle.dump(infos, f)
        return        