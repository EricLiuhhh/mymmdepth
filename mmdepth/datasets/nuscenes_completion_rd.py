import os
import glob
import pickle
from typing import List, Union
from mmengine.dataset import BaseDataset
from mmdepth.registry import DATASETS

@DATASETS.register_module()
class NuScenesCompletionRD(BaseDataset):
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
    @classmethod
    def create_data(self, split, root_path, out_path, pkl_prefix):
        if split == 'train':
            glob_data = os.path.join(
                root_path,
                'train/*.h5'
            )
        elif split == 'val' or split == 'test':
            glob_data = os.path.join(
                root_path,
                'val/*.h5'
            )
        paths_data = sorted(glob.glob(glob_data))
        items = []
        for i in range(len(paths_data)):
            item = {'h5_path': paths_data[i]}
            items.append(item)
        infos = {
            'metainfo': {},
            'data_list': items
        }
        with open(os.path.join(out_path, f'{pkl_prefix}_infos_{split}.pkl'), 'wb') as f:
            pickle.dump(infos, f)
        