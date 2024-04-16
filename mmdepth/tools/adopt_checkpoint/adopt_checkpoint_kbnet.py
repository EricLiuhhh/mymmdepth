import sys
sys.path.append('.')
from collections import OrderedDict
import re
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdepth.tools.checkpoint_matching import match_chackpoint

cfg_file = 'configs/kbnet/kbnet_kitti.py'
checkpoint_file = 'checkpoints/kbnet-kitti.pth'
save_file = 'checkpoints/kbnet_new.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)
#res = match_chackpoint(old_sd, new_sd, replacements={'lidar':'depth'}, fast=False, full_output=False)
adopted_sd = OrderedDict()

cnt = 0
# match prefiller
related_old = OrderedDict(old_sd['sparse_to_dense_pool_state_dict'].items())
related_new = OrderedDict(filter(lambda x: (x[0].startswith('prefiller')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd['sparse_to_dense_pool_state_dict'][old_key].shape
    adopted_sd[new_key] = old_sd['sparse_to_dense_pool_state_dict'][old_key]

# match img encoder
related_old = OrderedDict(filter(lambda x: ('image' in x[0]), old_sd['encoder_state_dict'].items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_branch')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()
val = related_old.pop('module.conv5_image.conv_block.0.conv.weight')
related_old['module.conv5_image.conv_block.0.conv.weight'] = val
for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd['encoder_state_dict'][old_key].shape
    adopted_sd[new_key] = old_sd['encoder_state_dict'][old_key]

# match depth encoder
related_old = OrderedDict(filter(lambda x: (re.search('conv[0-9]?.depth', x[0]) is not None), old_sd['encoder_state_dict'].items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('depth_branch')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

val = related_old.pop('module.conv5_depth.conv_block.0.conv.weight')
related_old['module.conv5_depth.conv_block.0.conv.weight'] = val
for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd['encoder_state_dict'][old_key].shape
    adopted_sd[new_key] = old_sd['encoder_state_dict'][old_key]

# match backproj
related_old = OrderedDict(filter(lambda x: ('proj_depth' in x[0] or 'conv_fused' in x[0]), old_sd['encoder_state_dict'].items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('backproj')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd['encoder_state_dict'][old_key].shape
    adopted_sd[new_key] = old_sd['encoder_state_dict'][old_key]

# match decoder
related_old = OrderedDict(filter(lambda x: (re.search('deconv[0-9].conv', x[0]) is not None), old_sd['decoder_state_dict'].items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('decoder.guides')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd['decoder_state_dict'][old_key].shape
    adopted_sd[new_key] = old_sd['decoder_state_dict'][old_key]

related_old = OrderedDict(filter(lambda x: (re.search('deconv[0-9].deconv', x[0]) is not None or 'output0' in x[0]), old_sd['decoder_state_dict'].items()))
related_new = OrderedDict(filter(lambda x: (x[0].startswith('decoder.layers')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd['decoder_state_dict'][old_key].shape
    adopted_sd[new_key] = old_sd['decoder_state_dict'][old_key]

runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
pass