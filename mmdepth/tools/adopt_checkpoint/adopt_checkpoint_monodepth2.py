import sys
sys.path.append('.')
import re
from collections import OrderedDict
import torch
from mmengine.config import Config
from mmengine.runner import Runner

cfg_file = 'configs/monodepth2/monodepth2_kitti.py'
checkpoint_file = 'checkpoints/mono_1024x320/encoder.pth'
save_file = 'checkpoints/monodepth2_new.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)
adopted_sd = OrderedDict()

cnt = 0
# match encoder
related_old = OrderedDict(filter(lambda x: (x[0].startswith('encoder') and not x[0].startswith('encoder.fc')), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_encoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

old_sd = torch.load('checkpoints/mono_1024x320/depth.pth')
# match decoder
related_old = OrderedDict(filter(lambda x: (re.search('decoder\.[1,3,5,7]\.conv', x[0]) is not None), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_decoder.guide')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match decoder
related_old = OrderedDict(filter(lambda x: (re.search('decoder\.[0,2,4,6,8,9]\.conv', x[0]) is not None), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_decoder') and 'layers' in x[0]), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match decoder
related_old = OrderedDict(filter(lambda x: (re.search('decoder\.[0-9][0-9]\.conv', x[0]) is not None), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_decoder') and 'conv_outs' in x[0]), new_sd.items()))
related_new = OrderedDict(sorted(related_new.items(), key=lambda x: x[0], reverse=True))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]


old_sd = torch.load('checkpoints/mono_1024x320/pose_encoder.pth')
# match pose encoder
related_old = OrderedDict(filter(lambda x: not x[0].startswith('encoder.fc'), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('pose_encoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

old_sd = torch.load('checkpoints/mono_1024x320/pose.pth')
related_old = OrderedDict(old_sd)
related_new = OrderedDict(filter(lambda x: (x[0].startswith('pose_decoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
pass