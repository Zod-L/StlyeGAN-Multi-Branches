import os, sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
import cv2

import point_cloud_utils as pcu
import torch
from torch.nn import functional as F

def save_obj_mesh_with_rgb(mesh_path, verts, faces, rgbs):
    file = open(mesh_path, 'w')
    num = 0
    for v in verts:
        file.write('v %.4f %.4f %.4f' % (v[0], v[1], v[2]))
        file.write(' %.4f %.4f %.4f\n' % (rgbs[0][num], rgbs[1][num], rgbs[2][num]))
        num += 1
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

def fetchFaceUV():
    ### 根据template查找 crop face的对应点    
    # v is a nv by 3 NumPy array of vertices

    # downsample croped face
    v, f_simp = pcu.load_mesh_vf("./global2mesh/data/meanShape1.ply") 
    # original croped face
    #v, f_simp = pcu.load_mesh_vf("data/meanShape_test.ply") 
    
    # original head topo
    vs, fs = pcu.load_mesh_vf("./global2mesh/data/meanShape.ply")

    # For each query point, find the closest point on the mesh.
    # Here:
    #  - d is an array of closest distances for each query point with shape (1000,)
    #  - fi is an array of closest face indices for each point with shape (1000,)
    #  - bc is an array of barycentric coordinates within each face (shape (1000, 3)
    #    of the closest point for each query point
    d, fi, bc = pcu.closest_points_on_mesh(v, vs, fs)
    d_inv, fi_inv, bc_inv = pcu.closest_points_on_mesh(vs, v, f_simp)

    ### 获取uv坐标
    predef = './global2mesh/data/'

    face_files = predef + 'predef_faces.pkl'
    face_info = pickle.load(open(face_files,'rb'))

    uv_files = predef + 'predef_texcoords.pkl'
    uv_info = pickle.load(open(uv_files,'rb'))

    uv_coords = np.array(uv_info)
    #print(uv_coords.shape)

    temp = []
    for f in face_info:
        temp.append(f[2])
        
    uv_faces = np.array(temp)-1
    #print(np.max(uv_faces))

    temp = []
    for f in face_info:
        temp.append(f[0])
        
    uv_mapping = np.array(temp)-1

    ### face_v 与 face_vt顺序不同，需要对准
    #* vt compare to v
    mapping = np.zeros(26404)  
    #* v compare to vt
    inv_mapping = np.zeros(vs.shape[0])
    for i in range(uv_mapping.shape[0]):
        for j in range(uv_mapping.shape[1]):
            mapping[uv_faces[i][j]] = uv_mapping[i][j]
            inv_mapping[uv_mapping[i][j]] = uv_faces[i][j]
            
    mapping = mapping.astype(np.int)
    inv_mapping = inv_mapping.astype(np.int)

    def process_uv(uv_coords, uv_h = 256, uv_w = 256):
        ### 将uv坐标转换到图像坐标
        # input: uv_coords[n,2]
        # output:uv_coords[n,3]

        uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
        uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
        uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
        return uv_coords

    size = 512
    uv_h = uv_w = size
    image_h = image_w = size  

    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    # 增大面部在uv图像中的面积
    uv_coords = (uv_coords-size/2) * 1.4 + np.array([uv_h*0.5,uv_h*0.625,0]) 

    wanted = uv_coords[inv_mapping,:]
    uv_coords = wanted
    #找到 crop 后面部的uv坐标
    closest_points = pcu.interpolate_barycentric_coords(fs, fi, bc, uv_coords) 
  
    return closest_points, f_simp



def global2mesh(out_path, text, loc, size=512):
    ### generating mesh from uv ###

    #输出位置
    # uv数据

    uv_coords, faces = fetchFaceUV()


    


    # 保存位置
    mesh_path = out_path + '.obj'
    #print(mesh_path)
    
    # 从uv中获取数值
    grid = torch.from_numpy(uv_coords[np.newaxis,:,:].astype('float32')).unsqueeze(2)/size # [B, N, 1, 2]
    grid = grid[:,:,:,:2] * 2 - 1
    temp = loc.unsqueeze(0).float()
    
    tex_temp = text.unsqueeze(0).float()/255
    loc = F.grid_sample(temp, grid=grid, mode='bilinear').squeeze()
    loc = loc.numpy().squeeze().T  
    #print(np.max(loc), np.min(loc))
    
    colors = F.grid_sample(tex_temp, grid=grid, mode='bilinear').squeeze()
    colors = colors.numpy().squeeze().T      

    save_obj_mesh_with_rgb(mesh_path, loc, faces, colors.T)


if __name__=='__main__':
    ### generating mesh from uv ###
    path = './test/'

    #输出位置
    out_path = path + 'mesh/'
    # uv数据
    path = path + 'out/' 

    files = os.listdir(path)

    size = 512
    uv_coords, faces = fetchFaceUV()
    for file in files:

        if file[-4:] == '.npy':
            print(file)
            temp = np.load(path + file) 
            
            tex = cv2.imread(path + file[:-4] + '.png')[:,:,[2,1,0]]

        else:
            continue

        # 保存位置
        mesh_path = out_path + file[:-4] + '.obj'
        print(mesh_path)
        
        # 从uv中获取数值
        grid = torch.from_numpy(uv_coords[np.newaxis,:,:].astype('float32')).unsqueeze(2)/size # [B, N, 1, 2]
        grid = grid[:,:,:,:2] * 2 - 1
        temp = torch.from_numpy(temp[np.newaxis,:,:,:].astype('float32')).permute(0,3,1,2)
        
        tex_temp = torch.from_numpy(tex[np.newaxis,:,:,:].astype('float32')).permute(0,3,1,2)/255
        loc = F.grid_sample(temp, grid=grid, mode='bilinear').squeeze()
        loc = loc.numpy().squeeze().T  
        #print(np.max(loc), np.min(loc))
        
        colors = F.grid_sample(tex_temp, grid=grid, mode='bilinear').squeeze()
        colors = colors.numpy().squeeze().T      

        save_obj_mesh_with_rgb(mesh_path, loc, faces, colors.T)
