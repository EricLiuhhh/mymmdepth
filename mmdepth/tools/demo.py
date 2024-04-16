import sys
sys.path.append('.')
import argparse
from functools import partial
import numpy as np
import torch
from mmengine import DefaultScope
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import FUNCTIONS
from mmdepth.visualization import DepthVisualizer
from mmdepth.registry import MODELS, VISUALIZERS

CONFIG='configs/nlspn/nlspn_kitti.py'
IMG_PATH='data/kitti_depth_esti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png'
DEP_PATH='data/kitti_depth_esti/2011_09_26/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png'
CKP_PATH='checkpoints/NLSPN_KITTI_DC_new.pt'
SAVE_PATH='.'

def parse_args():
    parser = argparse.ArgumentParser(description='Simple test a detector')
    parser.add_argument('--config', default=CONFIG),
    parser.add_argument('--ckpt_path', default=CKP_PATH),
    parser.add_argument('--img_path', default=IMG_PATH),
    parser.add_argument('--depth_path', default=DEP_PATH),
    parser.add_argument('--save_path', default=SAVE_PATH)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_info = {
        'img_path': args.img_path,
        'depth_path': args.depth_path
    }
    cfg = Config.fromfile(args.config)
    default_scope = DefaultScope.get_instance(  # type: ignore
                'demo',
                scope_name='mmdepth')
    
    data_pipeline = Compose(cfg.val_pipeline)
    cfg.visualizer.update(save_dir=args.save_path)
    vis: DepthVisualizer = VISUALIZERS.build(cfg.visualizer)
    collate_fn_cfg = cfg.val_dataloader.pop('collate_fn',
                                            dict(type='pseudo_collate'))
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = FUNCTIONS.get(collate_fn_type)
    collate_fn = partial(collate_fn, **collate_fn_cfg)
    model:torch.nn.Module = MODELS.build(cfg.model).cuda().eval()
    model.load_state_dict(torch.load(args.ckpt_path))
    data = data_pipeline(data_info)
    data_batch = collate_fn([data])
    with torch.no_grad():
        data_samples = model.val_step(data_batch)
    pred_depth = data_samples[0].pred_depth.data.cpu().numpy()
    vis.add_depth('z', np.squeeze(pred_depth))

if __name__ == '__main__':
   main() 