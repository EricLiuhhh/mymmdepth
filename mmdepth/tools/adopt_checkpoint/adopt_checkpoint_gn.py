import sys
sys.path.append('.')
from collections import OrderedDict
import re
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdepth.tools.adopt_checkpoint.checkpoint_matching import match_chackpoint

cfg_file = 'configs/guidenet/guidenet_kitti.py'
checkpoint_file = 'checkpoints/gns_kitti.pth'
save_file = 'checkpoints/guidenet_new.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)['net']
#res = match_chackpoint(old_sd, new_sd, replacements={'lidar':'depth'}, fast=False, full_output=False)
adopted_sd = OrderedDict()

# match img encoder
cnt = 0
related_old = OrderedDict(filter(lambda x: ((re.search('conv_img', x[0]) is not None) or (re.search('layer[0-9]_img', x[0]) is not None)), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_encoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()
for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match img decoder
for i in range(2, 6):
    related_old = OrderedDict(filter(lambda x: ((re.search(f'layer{i}d_img', x[0]) is not None)), old_sd.items()))
    related_new = OrderedDict(filter(lambda x: (x[0].startswith(f'img_decoder.layers.{5-i}')), new_sd.items()))
    assert related_old.__len__() == related_new.__len__()
    cnt += related_old.__len__()
    for new_key, old_key in zip(related_new.keys(), related_old.keys()):
        if hasattr(new_sd[new_key], 'shape'):
            assert new_sd[new_key].shape == old_sd[old_key].shape
        adopted_sd[new_key] = old_sd[old_key]

# match depth encoder
related_old = OrderedDict(filter(lambda x: ((re.search('conv_lidar', x[0]) is not None) or (re.search('layer[0-9]_lidar', x[0]) is not None)), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('depth_encoder') and 'guides' not in x[0]), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()
for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match guide
related_old = OrderedDict(filter(lambda x: ((re.search('guide', x[0]) is not None)), old_sd.items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('depth_encoder') and 'guides' in x[0]), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()
for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match decoder
for i in range(1, 6):
    related_old = OrderedDict(filter(lambda x: ((re.search(f'layer{i}d\.', x[0]) is not None)), old_sd.items()))
    related_new = OrderedDict(filter(lambda x: (x[0].startswith(f'decoder.layers.{5-i}')), new_sd.items()))
    assert related_old.__len__() == related_new.__len__()
    cnt += related_old.__len__()
    for new_key, old_key in zip(related_new.keys(), related_old.keys()):
        if hasattr(new_sd[new_key], 'shape'):
            assert new_sd[new_key].shape == old_sd[old_key].shape
        adopted_sd[new_key] = old_sd[old_key]

# match conv out
related_old = OrderedDict(filter(lambda x: ((re.search('ref', x[0]) is not None) or ('module.conv.' in x[0])), old_sd.items()))
v = related_old.pop('module.conv.weight')
related_old['module.conv.weight'] = v
v = related_old.pop('module.conv.bias')
related_old['module.conv.bias'] = v
related_new = OrderedDict(filter(lambda x: (x[0].startswith('conv_out')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()
for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

assert cnt == old_sd.__len__()
runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
pass