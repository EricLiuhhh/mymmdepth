import sys
sys.path.append('.')
import re
from collections import OrderedDict
import torch
from mmengine.config import Config
from mmengine.runner import Runner

cfg_file = 'configs/radar_depth/radar_depth_vod.py'
checkpoint_file = 'checkpoints/resnet18_latefusion.pth.tar'
save_file = 'checkpoints/radar_depth_stage1.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)['model_state_dict']
adopted_sd = OrderedDict()

cnt = 0
# match img encoder
related_old = OrderedDict(filter(lambda x: (re.search('^layer[0-9]\.', x[0]) is not None or (x[0].startswith('conv1.') or x[0].startswith('bn1.'))), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_encoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match depth encoder
related_old = OrderedDict(filter(lambda x: ('depth' in x[0]), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('depth_encoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match fusion module
related_old = OrderedDict(filter(lambda x: ('fusion' in x[0]) or (x[0].startswith('conv2.') or x[0].startswith('bn2.')), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('fusion_module')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match decoder
related_old = OrderedDict(filter(lambda x: ('decoder' in x[0]), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('decoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match conv out
related_old = OrderedDict(filter(lambda x: ('conv3' in x[0]), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('conv_out')), new_sd.items()))
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