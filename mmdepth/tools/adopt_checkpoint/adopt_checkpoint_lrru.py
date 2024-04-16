import sys
sys.path.append('.')
import re
from collections import OrderedDict
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdepth.tools.checkpoint_matching import match_chackpoint

cfg_file = 'configs/lrru/lrru_kitti.py'
checkpoint_file = 'checkpoints/LRRU_Base.pt'
save_file = 'checkpoints/lrru_base_new.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)
#res = match_chackpoint(old_sd, new_sd, replacements={'lidar':'depth'}, fast=False, full_output=False)
adopted_sd = OrderedDict()

# match img encoder
img_related_old = OrderedDict(filter(lambda x: ('img' in x[0]), old_sd.items()))
img_related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_encoder')), new_sd.items()))
assert img_related_old.__len__() == img_related_new.__len__()

for new_key, old_key in zip(img_related_new.keys(), img_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match depth encoder
depth_related_old = OrderedDict(filter(lambda x: ('lidar' in x[0]), old_sd.items()))
depth_related_new = OrderedDict(filter(lambda x: (x[0].startswith('depth_encoder') and 'guide' not in x[0]), new_sd.items())) 

for new_key, old_key in zip(depth_related_new.keys(), depth_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match decoder
decoder_related_old = OrderedDict(filter(lambda x: re.search('layer[0-9]d', x[0]) is not None, old_sd.items()))
decoder_related_new = OrderedDict(filter(lambda x: (x[0].startswith('decoder') and 'conv_out' not in x[0]), new_sd.items())) 

for new_key, old_key in zip(decoder_related_new.keys(), decoder_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match guide
guide_related_old = OrderedDict(filter(lambda x: 'guide' in x[0], old_sd.items()))
guide_related_new = OrderedDict(filter(lambda x: 'guide' in x[0], new_sd.items())) 

for new_key, old_key in zip(guide_related_new.keys(), guide_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match refinement
refinement_related_old = OrderedDict(filter(lambda x: ('upproj' in x[0]) or ('offset' in x[0]), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: x[0].startswith('refinement') and 'Post' not in x[0], new_sd.items())) 

for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# others
others_related_old = OrderedDict(filter(lambda x: x[0].startswith('conv.conv'), old_sd.items()))
others_related_new = OrderedDict(filter(lambda x: x[0].startswith('decoder') and 'conv_out' in x[0], new_sd.items())) 

for new_key, old_key in zip(others_related_new.keys(), others_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

adopted_sd['refinement.Post_process.w'] = old_sd['Post_process.w']
adopted_sd['refinement.Post_process.b'] = old_sd['Post_process.b']
runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
print(f'Successfully save new checkpoint to {save_file}')
pass