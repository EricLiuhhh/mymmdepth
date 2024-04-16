'''
data
    - kitti_raw_data
        - 2011_09_26
            - 2011_09_26_drive_*_sync
                - image_02
                - image_03
            - ...
            - calib_cam_to_cam.txt
        - 2011_09_27
            - ...
depth_selection
train
    - 2011_09_26_drive_*_sync
        - proj_depth
            - groundtruth/image_02(03)
            - velodyne_raw/image_02(03)

'''

import os
import os.path as osp
import shutil


def setup_kitti_comp_dataset(train_path, raw_data_path):
    seqs = os.listdir(train_path)
    for seq in seqs:
        date = seq[:10]
        raw_path = os.path.join(raw_data_path, date, seq)
        calib_path = os.path.join(train_path, seq, 'calibration')
        os.makedirs(calib_path, exist_ok=True)
        if not os.path.exists(os.path.join(calib_path, 'calib_cam_to_cam.txt')):
            shutil.copy(os.path.join(raw_data_path, date, 'calib_cam_to_cam.txt'), os.path.join(calib_path, 'calib_cam_to_cam.txt'))
        if not os.path.exists(os.path.join(train_path, seq, 'image_02')):
            os.symlink(os.path.join(raw_path, 'image_02'), os.path.join(train_path, seq, 'image_02'))
        if not os.path.exists(os.path.join(train_path, seq, 'image_03')):
            os.symlink(os.path.join(raw_path, 'image_03'), os.path.join(train_path, seq, 'image_03'))

def setup_kitti_esti_dataset(raw_data_path, ann_data_path):
    ann_train_path = osp.join(ann_data_path, 'train')
    ann_val_path = osp.join(ann_data_path, 'val')
    train_seqs = os.listdir(ann_train_path)
    val_seqs = os.listdir(ann_val_path)
    for seq in train_seqs:
        date = seq[:10]
        os.unlink(osp.join(raw_data_path, date, seq, 'proj_depth'))
        os.symlink(osp.join(ann_train_path, seq, 'proj_depth'), osp.join(raw_data_path, date, seq, 'proj_depth'))
    for seq in val_seqs:
        date = seq[:10]
        os.unlink(osp.join(raw_data_path, date, seq, 'proj_depth'))
        os.symlink(osp.join(ann_val_path, seq, 'proj_depth'), osp.join(raw_data_path, date, seq, 'proj_depth'))

def unsetup_kitti_comp_dataset(train_path):
    seqs = os.listdir(train_path)
    for seq in seqs:
        calib_path = os.path.join(train_path, seq, 'calibration')
        shutil.rmtree(calib_path)
        os.remove(os.path.join(train_path, seq, 'image_02'))
        os.remove(os.path.join(train_path, seq, 'image_03'))


if __name__ == '__main__':
    # train_path = '/dataset/kitti_depth/train'
    # raw_data_path = '/dataset/kitti_depth/data/kitti_raw_data'
    # setup_kitti_comp_dataset(train_path, raw_data_path)

    raw_data_path = 'data/kitti_depth_esti'
    ann_data_path = '/dataset/kitti_depth'
    setup_kitti_esti_dataset(raw_data_path, ann_data_path)