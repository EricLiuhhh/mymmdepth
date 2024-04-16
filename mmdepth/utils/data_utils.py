import numpy as np
from PIL import Image
import torch
from torch import Tensor

def save_depth_as_uint16png(img, filename):
    #from tensor
    if isinstance(img, Tensor):
        img = np.squeeze(img.data.cpu().numpy())
    elif isinstance(img, np.ndarray):
        img = np.squeeze(img)
    else:
        raise NotImplementedError
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)