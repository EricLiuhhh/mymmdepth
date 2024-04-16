import sys
sys.path.append('.')
import re
from collections import OrderedDict
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdepth.tools.checkpoint_matching import match_chackpoint

cfg_file = 'configs/lrru/lrru_kitti.py'
checkpoint_file = 'checkpoints/lrru_base_new.pt'
save_file = 'checkpoints/lrru_base_new_new.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)
#res = match_chackpoint(old_sd, new_sd, replacements={'lidar':'depth'}, fast=False, full_output=False)
adopted_sd = OrderedDict()

cnt = 0
# match img encoder
img_related_old = OrderedDict(filter(lambda x: (x[0].startswith('img_encoder')), old_sd.items()))
img_related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_encoder')), new_sd.items()))
assert img_related_old.__len__() == img_related_new.__len__()
cnt += img_related_old.__len__()

for new_key, old_key in zip(img_related_new.keys(), img_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# match depth encoder
depth_related_old = OrderedDict(filter(lambda x: (x[0].startswith('depth_encoder')), old_sd.items()))
depth_related_new = OrderedDict(filter(lambda x: (x[0].startswith('depth_encoder')), new_sd.items())) 

for new_key, old_key in zip(depth_related_new.keys(), depth_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]
cnt += depth_related_old.__len__()

# match decoder
decoder_related_old = OrderedDict(filter(lambda x: (x[0].startswith('decoder')), old_sd.items()))
decoder_related_new = OrderedDict(filter(lambda x: (x[0].startswith('decoder')), new_sd.items())) 

for new_key, old_key in zip(decoder_related_new.keys(), decoder_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]
cnt += decoder_related_old.__len__()

# match refinement
refinement_related_old = OrderedDict(filter(lambda x: x[0].startswith('refinement'), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: x[0].startswith('refinement'), new_sd.items())) 

for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]
cnt += refinement_related_old.__len__()

assert cnt == old_sd.__len__() == new_sd.__len__()

runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
print(f'Successfully save new checkpoint to {save_file}')
pass