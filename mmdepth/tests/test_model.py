import sys
sys.path.append('.')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from mmengine.config import Config
from mmengine.runner import Runner

cfg_path = 'configs/gaussian_depth/gaussian_depth_vod.py'
#ckp_path = 'exps/temp/epoch_20.pth'
ckp_path = 'checkpoints/radar_depth_stage1_nuscenes.pt'

cfg = Config.fromfile(cfg_path)
cfg.work_dir = 'exps/temp'
#cfg.load_from = ckp_path
runner = Runner.from_cfg(cfg)
#runner.model.show_network()
runner.train()
#runner.train()

