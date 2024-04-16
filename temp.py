import pickle
with open('data/kitti_depth/kitti_completion_infos_train.pkl', 'rb') as f:
    infos = pickle.load(f)
pass

# import random
# import time
# import torch
# from PENet_ICRA2021.CoordConv import AddCoordsNp
# from mmdepth.models.layers.positional_encoding import PositionalEncoding

# s=time.time()
# for i in range(10):
#     xdim = random.randint(1000, 2000)
#     ydim = random.randint(300, 600)
#     a = AddCoordsNp(ydim, xdim)
#     res1 = a.call()

#     mask = torch.zeros((1, ydim, xdim), dtype=torch.bool)
#     a=PositionalEncoding(scale=2, offset=-1)
#     res2 = a(mask)[0]

#     print((res1-res2.numpy()).sum())
# print(time.time()-s)



# from PIL import Image
# import numpy as np

# dep = np.array(Image.open('data/kitti_depth/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png'))
# pass