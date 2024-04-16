seqs = (
    (0, 543),
    (544, 1311),
    (1312, 1802),
    (1803, 2199),
    (2200, 2531),
    (2532, 2797),
    (2798, 3276),
    (3277, 3574),
    (3575, 3609),
    (3610, 4047),
    (4049, 4386),
    (4387, 4651),
    (4652, 5085),
    (6334, 6570),
    (6571, 6758),
    (6759, 7542),
    (7543, 7899),
    (7900, 8197),
    (8198, 8480),
    (8481, 8748),
    (8749, 9095),
    (9096, 9517),
    (9518, 9775),
    (9776, 9930)
)

import os 
import os.path as osp
import shutil
from tqdm import tqdm
data_root = './data/vod_raw'
save_path = './data/vod_depth_esti'
os.makedirs(save_path, exist_ok=True)

for i, seq in enumerate(tqdm(seqs)):
    seq_id = f'{i:02d}'
    frame_ids = [f'{j:05d}' for j in range(seq[0], seq[1]+1)]
    target_img_dir = osp.join(save_path, seq_id, 'image_02')
    point_types = ['lidar', 'radar', 'radar_3frames', 'radar_5frames']
    target_point_dirs = [osp.join(save_path, seq_id, point_type) for point_type in point_types]
    target_calib_dirs = [osp.join(save_path, seq_id, 'calib', point_type) for point_type in point_types]
    # image
    os.makedirs(target_img_dir, exist_ok=True)
    for frame_id in frame_ids:
        shutil.copy(osp.join(data_root, 'lidar', 'training', 'image_2', frame_id+'.jpg'),
                    osp.join(target_img_dir, frame_id+'.jpg'))
        
    # point
    for point_type, point_dir in zip(point_types, target_point_dirs):
        os.makedirs(point_dir, exist_ok=True)
        for frame_id in frame_ids:
            shutil.copy(osp.join(data_root, point_type, 'training', 'velodyne', frame_id+'.bin'),
                    osp.join(point_dir, frame_id+'.bin'))
    
    # calib
    for point_type, calib_dir in zip(point_types, target_calib_dirs):
        os.makedirs(calib_dir, exist_ok=True)
        for frame_id in frame_ids:
            shutil.copy(osp.join(data_root, point_type, 'training', 'calib', frame_id+'.txt'),
                    osp.join(calib_dir, frame_id+'.txt'))