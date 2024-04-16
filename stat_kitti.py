import os
paths = os.listdir('data/kitti_depth_esti')
paths = sorted([p for p in paths if p.startswith('2011')])
cnt = 0
all_seqs = []
for p in paths:
    seqs = os.listdir(os.path.join('data/kitti_depth_esti', p))
    seqs = sorted([s for s in seqs if s.startswith('2011')])
    cnt += len(seqs)
    all_seqs.extend(seqs)

split_file = '/home/lhs/depth_est_comp/monodepth2/splits/eigen_full/train_files.txt'
#split_file = '/home/lhs/monodepth/NeWCRFs/data_splits/eigen_train_files_with_gt.txt'
#split_file = '/home/lhs/depth_est_comp/monodepth2/splits/eigen/test_files.txt'
#split_file = '/home/lhs/monodepth/NeWCRFs/data_splits/kitti_depth_prediction_train.txt'

with open(split_file, 'r') as f:
    lines = f.readlines()
    
#lines = sorted(os.listdir('data/kitti_depth/train'))

lines = [x.split(' ')[0] for x in lines]
seqs = [x.split('/')[1] for x in lines]
#seqs = lines
seqs = (set(seqs))
# print(len(lines))
# print(len(seqs))

#split_file = '/home/lhs/depth_est_comp/monodepth2/splits/eigen_full/val_files.txt'
split_file = '/home/lhs/monodepth/NeWCRFs/data_splits/eigen_test_files_with_gt.txt'
#split_file = '/home/lhs/monodepth/NeWCRFs/data_splits/eigen_train_files_with_gt.txt'
#split_file = '/home/lhs/monodepth/NeWCRFs/data_splits/kitti_official_valid.txt'

with open(split_file, 'r') as f:
    lines2 = f.readlines()

#lines2 = sorted(os.listdir('data/kitti_depth/val'))

lines2 = [x.split(' ')[0] for x in lines2]
seqs2 = [x.split('/')[1] for x in lines2]
#seqs2 = lines2
seqs2 = (set(seqs2))
# print(len(lines2))
# print(len(seqs2))
# print(seqs.intersection(seqs2))

print('seq_len1:', len(seqs), 'frame_len1:', len(lines))
print('seq_len2:', len(seqs2), 'frame_len2:', len(lines2))
print('intersection:', seqs.intersection(seqs2))
for i, seq in enumerate(all_seqs):
    print(f'{i+1:3d}', seq, 'x' if seq in seqs else ' ', 'o' if seq in seqs2 else ' ')