import sys
sys.path.append('.')
import re
from collections import OrderedDict
import torch
from mmengine.config import Config
from mmengine.runner import Runner

cfg_file = 'configs/hrdepth/hrdepth_kitti.py'
checkpoint_file = 'checkpoints/HR_Depth_K_M_1280x384/encoder.pth'
save_file = 'checkpoints/hrdepth_new.pt'

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

old_sd = torch.load('checkpoints/HR_Depth_K_M_1280x384/depth.pth')
# match decoder
related_old = old_sd
related_new = OrderedDict(filter(lambda x: (x[0].startswith('img_decoder')), new_sd.items()))
assert related_old.__len__() == related_new.__len__()
cnt += related_old.__len__()

for new_key, old_key in zip(related_new.keys(), related_old.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

adopted_sd['backproject_depth.id_coords'] = new_sd['backproject_depth.id_coords']
adopted_sd['backproject_depth.ones'] = new_sd['backproject_depth.ones']
adopted_sd['backproject_depth.pix_coords'] = new_sd['backproject_depth.pix_coords']

runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
pass