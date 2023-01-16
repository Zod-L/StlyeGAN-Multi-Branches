import os, sys
import numpy as np
import pickle
from em2mesh.mesh import load_obj_mesh, save_obj_mesh

#import face3d
#from face3d import mesh
import matplotlib.pyplot as plt
import cv2

import point_cloud_utils as pcu
import torch
from torch.nn import functional as F


def scale_uv(uv_coords):
    temp = uv_coords * 2 - 1
    print(np.min(temp), np.max(temp))
    #temp = uv_coords * 2 - 1
    temp = np.square(temp)
    temp = np.sum(temp, 1)
    print(np.min(temp), np.max(temp))
    temp = 1.414 - np.sqrt(temp)
    temp = np.sqrt(temp)
    
    uv_coords = 1.5*temp*(uv_coords*2-1).T
    uv_coords = (uv_coords * 0.5 + 0.5)
    
    return uv_coords.T

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    #uv_coords = scale_uv(uv_coords)

    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.array(verts), np.array(faces)

def em2mesh(out_path, text, loc): 
    uv_temp = './em2mesh/utils/meanShape_vt_simp' ## template for local topo

    v, f_simp = pcu.load_mesh_vf("./em2mesh/meanShape1_shrinked2.ply")
    vs, fs = pcu.load_mesh_vf("./em2mesh/meanShape.ply")

    vs_strech, fs_strech = pcu.load_mesh_vf(uv_temp + ".ply")

    # For each query point, find the closest point on the mesh.
    # Here:
    #  - d is an array of closest distances for each query point with shape (1000,)
    #  - fi is an array of closest face indices for each point with shape (1000,)
    #  - bc is an array of barycentric coordinates within each face (shape (1000, 3)
    #    of the closest point for each query point
    ##d, fi, bc = pcu.closest_points_on_mesh(v, vs, fs)


    #    of the closest point for each query point
    d, fi, bc = pcu.closest_points_on_mesh(v, vs, fs)

    d_strech1, fi_strech1, bc_strech1 = pcu.closest_points_on_mesh(vs_strech, vs, fs)
    d_strech, fi_strech, bc_strech = pcu.closest_points_on_mesh(vs, vs_strech, fs_strech)

    # Convert barycentric coordinates to 3D positions
    closest_points = pcu.interpolate_barycentric_coords(fs, fi, bc, vs)


    #print(closest_points)

    #pcu.save_triangle_mesh("uv_remesh.ply", v=closest_points, f=f_simp)
    #exit()



    ## uv for strech mapping
    _, _, uv_coords_strech, uv_faces_strech, = load_obj_mesh(uv_temp + '.obj', with_texture=True)
    uv_coords = uv_coords_strech
    uv_faces = uv_faces_strech



    size = 512
    uv_h = uv_w = size
    image_h = image_w = size  


    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    uv_coords = (uv_coords-size*0.5) * 1.0 + np.array([size*0.5,size*0.5,0]) # scale for local topo

    ## the mapping for v, vt in mesh strech
    uv_mapping = fs_strech

    ### vt compare to v
    mapping = np.zeros(uv_coords_strech.shape[0])

        
    ### v compare to vt
    inv_mapping = np.zeros(vs_strech.shape[0])
    for i in range(uv_mapping.shape[0]):
        for j in range(uv_mapping.shape[1]):
            mapping[uv_faces[i][j]] = uv_mapping[i][j]
            inv_mapping[uv_mapping[i][j]] = uv_faces[i][j]
            
    mapping = mapping.astype(np.int)
    inv_mapping = inv_mapping.astype(np.int)

    uv_coords = uv_coords[inv_mapping]

    _, mask_face = pcu.load_mesh_vf("em2mesh/uv_temp.ply")

    #closest_points = pcu.interpolate_barycentric_coords(uv_faces, fi, bc, uv_coords)
    closest_points = pcu.interpolate_barycentric_coords(fs_strech, fi_strech, bc_strech, uv_coords)


    #exit()

    ### generating mesh from uv ###






    loc_max, loc_min = 150, -150





    out_path += ".obj"
        
    grid = torch.from_numpy(uv_coords[np.newaxis,:,:].astype('float32')).unsqueeze(2)/size # [B, N, 1, 2]
    grid = grid[:,:,:,:2] * 2 - 1

    loc = F.grid_sample(loc.unsqueeze(0), grid=grid, mode='bilinear').squeeze()
    loc = loc.numpy().squeeze().T

    save_obj_mesh(out_path, loc, fs_strech)
