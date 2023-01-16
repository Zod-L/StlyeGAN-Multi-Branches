from torch_utils.ops.grid_sample_gradfix import grid_sample
import numpy as np
import torch
import cv2
from torchvision.utils import save_image



temp1 = np.load("data/fuse/local/00000/img00000000.npy")
temp2 = np.load("data/fuse/local/00000/img00000114.npy")
temp1 = torch.from_numpy(temp1[np.newaxis,:,:,:].astype('float32')).permute(0,3,1,2) / 127.5 - 1
temp2 = torch.from_numpy(temp2[np.newaxis,:,:,:].astype('float32')).permute(0,3,1,2) / 127.5 - 1
img = torch.cat((temp1, temp2), dim=0)
grid = torch.from_numpy(np.load("local_to_global/grid_global_sample.npy"))[:,:,:,:2] * 2 - 1
grid = grid.repeat((img.shape[0], 1, 1, 1))
mask = torch.from_numpy(np.load("local_to_global/grid_global_mask.npy")).unsqueeze(0).permute(0, 3, 1, 2)
mask = mask.repeat((img.shape[0], 1, 1, 1))
img = grid_sample(img, grid)
img = (img + 1) * mask - 1
save_image(img, "1.png", normalize=True, range=(-1, 1))
