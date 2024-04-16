import os
import glob
import pickle
from typing import List, Union
from mmengine.dataset import BaseDataset
from mmdepth.registry import DATASETS

@DATASETS.register_module()
class KittiCompletion(BaseDataset):
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
    def create_data(self, split, root_path, out_path, pkl_prefix, split_file=None):
        if split_file is not None:
            with open(split_file, 'r') as f:
                lines = f.readlines()
            lines = [line.strip().split(' ') for line in lines]
            items = []
            for line in lines:
                date = line[0].split('/')[0]
                seq = line[0].split('/')[1]
                frame_idx = line[1]
                item = {
                # ann dir
                'depth_path': os.path.join(root_path, date, seq, 'proj_depth', 'velodyne_raw', f'image_0{self.side_map[line[-1]]}', f'{int(frame_idx):010d}.png'),
                'depth_gt_path': os.path.join(root_path, date, seq, 'proj_depth', 'groundtruth', f'image_0{self.side_map[line[-1]]}', f'{int(frame_idx):010d}.png'),
                # raw data dir
                'velodyne_path': os.path.join(root_path, date, seq, 'velodyne_points', 'data', f'{int(frame_idx):010d}.bin'),
                'calib_path': os.path.join(root_path, date, 'calib_cam_to_cam.txt'),
                'calib_velo_path': os.path.join(root_path, date, 'calib_velo_to_cam.txt'),
                'img_path': os.path.join(root_path, date, seq, f'image_0{self.side_map[line[-1]]}', 'data', f'{int(frame_idx):010d}.png'),
                'side': line[-1]}
                items.append(item)
            infos = {
                'metainfo': {},
                'data_list': items
            }
            with open(os.path.join(out_path, f'{pkl_prefix}_infos_{split}.pkl'), 'wb') as f:
                pickle.dump(infos, f)
            return

        if split == 'train':
            glob_dep = os.path.join(
                root_path,
                'train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                root_path,
                'train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                if 'image_02' in p:
                    tmp = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
                elif 'image_03' in p:
                    tmp = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
                else:
                    raise ValueError('ERROR')
                return tmp
            def get_K_paths(p):
                return p.split('proj_depth')[0] + 'calibration/calib_cam_to_cam.txt'
        elif split == 'val_full':
            glob_dep = os.path.join(
                root_path,
                'val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                root_path,
                'val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                if 'image_02' in p:
                    tmp = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
                elif 'image_03' in p:
                    tmp = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
                else:
                    raise ValueError('ERROR')
                return tmp
            def get_K_paths(p):
                return p.split('proj_depth')[0] + 'calibration/calib_cam_to_cam.txt'
        elif split == 'val_select':
            glob_dep = os.path.join(
                root_path,
                'depth_selection/val_selection_cropped/velodyne_raw/*.png'
            )
            glob_gt = os.path.join(
                root_path,
                'depth_selection/val_selection_cropped/groundtruth_depth/*.png'
            )
            glob_K = os.path.join(
                root_path,
                'depth_selection/val_selection_cropped/intrinsics/*.txt'
            )
            glob_rgb = os.path.join(
                root_path,
                'depth_selection/val_selection_cropped/image/*.png'
            )
        elif split == "test_completion":

            glob_dep = os.path.join(
                root_path,
                'depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png'
            )
            glob_gt = None
            glob_K = os.path.join(
                root_path,
                'depth_selection/test_depth_completion_anonymous/intrinsics/*.txt'
            )
            glob_rgb = os.path.join(
                root_path,
                'depth_selection/test_depth_completion_anonymous/image/*.png'
            )
        elif split == "test_prediction":

            glob_dep, glob_gt = None, None
            glob_K = os.path.join(
                root_path,
                'depth_selection/test_depth_prediction_anonymous/intrinsics/*.txt'
            )
            glob_rgb = os.path.join(
                root_path,
                'depth_selection/test_depth_prediction_anonymous/image/*.png'
            )
        else:
            raise ValueError("Unrecognized split " + str(split))

        if glob_gt is not None:
            # train or val-full or val-select
            paths_dep = sorted(glob.glob(glob_dep))
            paths_gt = sorted(glob.glob(glob_gt))
            if split == 'train' or (split == 'val_full'):
                paths_rgb = [get_rgb_paths(p) for p in paths_dep]
                paths_K = [get_K_paths(p) for p in paths_dep]
            else:
                paths_rgb = sorted(glob.glob(glob_rgb))
                paths_K = sorted(glob.glob(glob_K))
        else:
            # test only has dep or rgb
            paths_K = sorted(glob.glob(glob_K))
            paths_rgb = sorted(glob.glob(glob_rgb))
            if split == "test_prediction":
                paths_dep = [None] * len(paths_rgb)  # test_prediction has no sparse depth
            else:
                paths_dep = sorted(glob.glob(glob_dep))
            paths_gt = [None] * len(paths_dep)

        if len(paths_dep) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0 and len(paths_K) == 0:
            raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
        if len(paths_dep) == 0:
            raise (RuntimeError("Requested sparse depth but none was found"))
        if len(paths_rgb) == 0:
            raise (RuntimeError("Requested rgb images but none was found"))
        if len(paths_rgb) == 0:
            raise (RuntimeError("Requested gray images but no rgb was found"))
        if len(paths_K) == 0:
            raise (RuntimeError("Requested structure images but no structure was found"))

        if len(paths_rgb) != len(paths_dep) or len(paths_rgb) != len(paths_gt) or len(paths_gt) != len(paths_K):
            print('Warning:', len(paths_dep), len(paths_gt), len(paths_rgb), len(paths_K))

        items = []
        for i in range(len(paths_rgb)):
            item = {'depth_path': paths_dep[i],
                'depth_gt_path': paths_gt[i],
                'calib_path': paths_K[i],
                'img_path': paths_rgb[i]}
            items.append(item)
        infos = {
            'metainfo': {},
            'data_list': items
        }
        with open(os.path.join(out_path, f'{pkl_prefix}_infos_{split}.pkl'), 'wb') as f:
            pickle.dump(infos, f)
        