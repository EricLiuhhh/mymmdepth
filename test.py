import argparse
from os import path as osp
from mmdepth.datasets import KittiCompletion
parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', type=str, default='kitti_completion', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti_depth',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti_depth',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti_completion')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti_completion':
        KittiCompletion.create_data('val_select', args.root_path, args.out_dir, args.extra_tag)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')