import sys
sys.path.append('.')
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from PIL import Image
from mmdepth.datasets.transforms.utils import generate_depth_map_vod, read_calib_file_vod
from mmdepth.utils.data_utils import save_depth_as_uint16png
from mmdepth.visualization.utils import depth2pcl

def _read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

data_root = 'data/vod_depth_esti'
seqs = os.listdir(data_root)
seqs = sorted([s for s in seqs if len(s)==2])
point_types = ['lidar', 'radar', 'radar_3frames', 'radar_5frames']
# point_types = ['radar_5frames']
for t in point_types:
    for seq in tqdm(seqs):
        calib_root = osp.join(data_root, seq, 'calib', t)
        pt_root = osp.join(data_root, seq, t)
        save_path = osp.join(data_root, seq, 'proj_depth', t)
        os.makedirs(save_path, exist_ok=True)
        calibs = sorted(os.listdir(calib_root))
        pts = sorted(os.listdir(pt_root))
        for calib, pt in zip(calibs, pts):
            frame_id = int(pt.split('.')[0])
            assert frame_id == int(calib.split('.')[0])
            depth_map = generate_depth_map_vod(osp.join(calib_root, calib), osp.join(pt_root, pt), pts_type=t)
            depth_map[depth_map>100.0] = 0.0
            save_depth_as_uint16png(depth_map, osp.join(save_path, f'{frame_id:05d}.png'))
            
            # check
            intrinsic, _ = read_calib_file_vod(osp.join(calib_root, calib))
            _depth = _read_depth(osp.join(save_path, f'{frame_id:05d}.png'))
            _pts: np.ndarray = depth2pcl(_depth, intrinsic[:3, :3], valid_map=(_depth>1e-4))
            _pts.astype(np.float32).tofile('z.bin')

