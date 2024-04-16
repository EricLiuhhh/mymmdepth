import sys
sys.path.append('.')
from collections import OrderedDict
import torch
from mmengine.config import Config
from mmengine.runner import Runner

cfg_file = 'configs/cformer/cformer_kitti.py'
checkpoint_file = 'checkpoints/cformer.pt'
save_file = 'checkpoints/cformer_new.pt'

cfg = Config.fromfile(cfg_file)
cfg.work_dir = 'exps/temp'
runner = Runner.from_cfg(cfg)
new_sd = runner.model.state_dict()
old_sd = torch.load(checkpoint_file)['net']
#res = match_chackpoint(old_sd, new_sd, replacements={'lidar':'depth'}, fast=False, full_output=False)
adopted_sd = OrderedDict()



# encoder3 = old_sd.pop('encoder3')
# encoder5 = old_sd.pop('encoder5')
# encoder7 = old_sd.pop('encoder7')
# old_keys = list(old_sd.keys()).copy()
# for k in old_keys:
#     if k.startswith('dimhalf') or k.startswith('att_12'):
#         old_sd.pop(k)
# new_sd.pop('refinement.trans_kernel3')
# new_sd.pop('refinement.trans_kernel5')
# new_sd.pop('refinement.trans_kernel7')

assert new_sd.__len__() == old_sd.__len__()
for new_key, old_key in zip(new_sd.keys(), old_sd.keys()):
    if hasattr(new_sd[new_key], 'shape'):
        assert new_sd[new_key].shape == old_sd[old_key].shape
    adopted_sd[new_key] = old_sd[old_key]

# adopted_sd['refinement.trans_kernel3'] = encoder3
# adopted_sd['refinement.trans_kernel5'] = encoder5
# adopted_sd['refinement.trans_kernel7'] = encoder7

runner.model.load_state_dict(adopted_sd, strict=True)
print('All keys matched!')
torch.save(adopted_sd, save_file)
pass