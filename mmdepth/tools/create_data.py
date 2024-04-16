import sys
sys.path.append('.')
import argparse
from os import path as osp
from mmdepth.datasets import KittiCompletion, VoDCompletion, NuScenesCompletionRD

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', type=str, default='vod_completion', help='name of the dataset')
parser.add_argument('--split', type=str, default='test')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/vod_depth_esti',
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
    default='./data/vod_depth_esti',
    required=False,
    help='name of info pkl')
parser.add_argument(
    '--split-file',
    type=str,
    required=False
)
parser.add_argument(
    '--point-type',
    type=str,
    default='radar_5frames',
    help='for VoD dataset'
)
parser.add_argument('--extra-tag', type=str, default='vod_completion')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti_completion' or args.dataset == 'kitti_estimation':
        args.split_file = '/home/lhs/depth_est_comp/monodepth2/splits/eigen/test_files.txt'
        KittiCompletion.create_data('eigen_test', args.root_path, args.out_dir, args.extra_tag, args.split_file)
    elif args.dataset == 'vod_completion' or args.dataset == 'vod_estimation':
        if args.split_file is None:
            args.split_file = osp.join(args.root_path, 'split', f'depth_split_{args.split}.txt')
        VoDCompletion.create_data(args.split, args.root_path, args.out_dir, args.extra_tag, args.split_file, args.point_type)
    elif args.dataset == 'nuscenes_completion_rd':
        NuScenesCompletionRD.create_data(args.split, args.root_path, args.out_dir, args.extra_tag)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')