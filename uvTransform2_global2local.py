import os, sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
import cv2

import torch
from torch.nn import functional as F


name = ['carishop_more/', 'caricshop/', 'facescape_neutral/', 'cartoon/', 'toonme/']
name = 'facescape_neutral/'

path = r'D:\Project\TopoTrans\textureloc/' # path for global locationmap
path = path + name

files = os.listdir(path)

out_path = 'D:/Project/TopoTrans/uv_data_stretch_onepiece/'
out_path = out_path + name

to_path = r'uv_data_stretch_onepiece\utils/' # predined info

size = 512
if not os.path.exists(out_path):
    os.mkdir(out_path)

for file in files:

    if file[-4:] == '.npy':
        print(file)
        temp = np.load(path + file)


        if np.max(temp)<100:                    
            temp = (loc_max-loc_min) * temp + (loc_min)
        print('=== temp:', np.max(temp),np.min(temp))
        if temp.shape[0]<size:
            continue
    else:
        continue
    
    name = out_path + file[:-4]
    mesh_path = out_path + file[:-4] + '.obj'

    temp = torch.from_numpy(temp[np.newaxis,:,:,:].astype('float32')).permute(0,3,1,2)#/280

    uv_sample_map = np.load(to_path + 'grid_local_sample_512.0.npy')
    grid2 = torch.from_numpy(uv_sample_map) 
    grid2 = grid2[:,:,:,:2] * 2 - 1
    
    loc2 = F.grid_sample(temp, grid=grid2, mode='bilinear').squeeze()
    loc2 = loc2.permute(1,2,0).numpy()/255.0
    
    mask = np.load(to_path + 'grid_mask.npy')
    
    
    #cv2.imshow('test', np.hstack([uv_sample_map,loc2, uv_location_map/255.0, loc2 - uv_location_map/255.0]))
    cv2.imshow('test', loc2 * mask)
    cv2.waitKey()
    #exit()
