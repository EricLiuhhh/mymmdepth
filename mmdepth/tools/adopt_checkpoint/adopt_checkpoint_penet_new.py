import sys
sys.path.append('.')
import re
from collections import OrderedDict
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdepth.tools.checkpoint_matching import match_chackpoint

cfg_file = 'configs/penet/penet_kitti.py'
checkpoint_file = 'checkpoints/penet_new.pt'
save_file = 'checkpoints/penet_new_new.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)
#res = match_chackpoint(old_sd, new_sd, replacements={'lidar':'depth'}, fast=False, full_output=False)
adopted_sd = OrderedDict()

cnt = 0
# img branch
img_related_old = OrderedDict(filter(lambda x: (x[0].startswith('img_branch')), old_sd.items()))
img_related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_branch')), new_sd.items()))
assert img_related_old.__len__() == img_related_new.__len__()
cnt += len(img_related_new)

for new_key, old_key in zip(img_related_new.keys(), img_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# depth branch
depth_related_old = OrderedDict(filter(lambda x: (x[0].startswith('depth_branch')), old_sd.items()))
depth_related_new = OrderedDict(filter(lambda x: (x[0].startswith('depth_branch')), new_sd.items()))
assert depth_related_old.__len__() == depth_related_new.__len__()
cnt += len(depth_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(depth_related_new.keys(), depth_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]


# refinement kernel_cnf branch
refinement_related_old = OrderedDict(filter(lambda x: ('kernel_conf_layer_s2' in x[0]), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: ('kernel_conf_layers.0' in x[0]), new_sd.items()))
assert refinement_related_old.__len__() == refinement_related_new.__len__()
cnt += len(refinement_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

refinement_related_old = OrderedDict(filter(lambda x: ('kernel_conf_layer' in x[0] and 's2' not in x[0]), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: ('kernel_conf_layers.1' in x[0]), new_sd.items()))
assert refinement_related_old.__len__() == refinement_related_new.__len__()
cnt += len(refinement_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

refinement_related_old = OrderedDict(filter(lambda x: ('mask_layer_s2' in x[0]), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: ('depth_conf_layers.0' in x[0]), new_sd.items()))
assert refinement_related_old.__len__() == refinement_related_new.__len__()
cnt += len(refinement_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]


refinement_related_old = OrderedDict(filter(lambda x: ('mask_layer' in x[0] and 's2' not in x[0]), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: ('depth_conf_layers.1' in x[0]), new_sd.items()))
assert refinement_related_old.__len__() == refinement_related_new.__len__()
cnt += len(refinement_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

refinement_related_old = OrderedDict(filter(lambda x: (re.search('iter_guide_layer[0-9]_s2', x[0]) is not None), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: ('affinity_layers.0' in x[0]), new_sd.items()))
assert refinement_related_old.__len__() == refinement_related_new.__len__()
cnt += len(refinement_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

refinement_related_old = OrderedDict(filter(lambda x: ('iter_guide_layer' in x[0] and 's2' not in x[0]), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: ('affinity_layers.1' in x[0]), new_sd.items()))
assert refinement_related_old.__len__() == refinement_related_new.__len__()
cnt += len(refinement_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

refinement_related_old = OrderedDict(filter(lambda x: ('trans_kernel' in x[0]), old_sd.items()))
refinement_related_new = OrderedDict(filter(lambda x: ('trans_kernel' in x[0]), new_sd.items()))
assert refinement_related_old.__len__() == refinement_related_new.__len__()
cnt += len(refinement_related_new)

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(refinement_related_new.keys(), refinement_related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
pass