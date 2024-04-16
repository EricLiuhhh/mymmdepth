import os
import os.path as osp
import numpy as np
np.random.seed(0)
def seq_len(data_root, seq):
    return len(list(filter(lambda x: x.endswith('jpg'), os.listdir(osp.join(data_root, f'{seq:02d}', 'image_02')))))

data_root = 'data/vod_depth_esti'
det_split = osp.join(data_root, 'split', 'det_split.txt')
with open(det_split, 'r') as f:
    splits = f.readlines()
splits = [s.strip() for s in splits]
all_seqs = set(range(len(splits)))
test_splits_det = [i for i, s in enumerate(splits) if s=='test']
print(test_splits_det)
# res = np.random.choice(test_splits_det, 3, replace=False)
res = set([5, 6, 21])
cnt = 0
for i in res:
    cnt += seq_len(data_root, i)
print(cnt)

test_split_depth = []
for i in res:
    samples = sorted(list(filter(lambda x: x.endswith('jpg'), os.listdir(osp.join(data_root, f'{i:02d}', 'image_02')))))
    for sample in samples:
        sample_id = int(sample.split('.')[0])
        test_split_depth.append(' '.join([f'{i:02d}', str(sample_id), 'l\n']))
np.random.shuffle(test_split_depth)
with open(osp.join(data_root, 'split', 'depth_split_test.txt'), 'w') as f:
    f.writelines(test_split_depth)

train_val_split_depth = []
train_val = all_seqs-res
cnt = 0
for i in train_val:
    cnt += seq_len(data_root, i)
print(cnt)
for i in train_val:
    samples = sorted(list(filter(lambda x: x.endswith('jpg'), os.listdir(osp.join(data_root, f'{i:02d}', 'image_02')))))
    for sample in samples:
        sample_id = int(sample.split('.')[0])
        train_val_split_depth.append(' '.join([f'{i:02d}', str(sample_id), 'l\n']))
np.random.shuffle(train_val_split_depth)
with open(osp.join(data_root, 'split', 'depth_split_trainval.txt'), 'w') as f:
    f.writelines(train_val_split_depth)