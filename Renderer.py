import os
import torch
from torch.nn import functional as F
import cv2

import numpy as np

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import pytorch3d
from local2mesh.mesh import load_obj_mesh
from torch_utils.ops import grid_sample_gradfix

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    AmbientLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftGouraudShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex
)

def compute_rotation(angles, device):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

def transform(face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans
        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

class uvOpt: # Create a uv set for specific topo

    def __init__(self, template, device, batch):
        vs, faces, uvs, uvfaces = load_obj_mesh(template,with_texture=True)
        size = uvs.shape[1]

        self.uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
        self.uvfaces = torch.tensor(uvfaces, dtype=torch.int64, device=device)
        self.faces = torch.tensor(faces, dtype=torch.int64, device=device)
        ## align uvs to verts
        '''
        mapping = torch.zeros(uvs.shape[0])
        inv_mapping = torch.zeros(vs.shape[0])
        for i in range(faces.shape[0]):
            for j in range(faces.shape[1]):
                mapping[uvfaces[i][j]] = faces[i][j]
                inv_mapping[faces[i][j]] = uvfaces[i][j]

        inv_mapping = inv_mapping.long()

        self.grid = (self.uvs[None,inv_mapping,:]*2-1).unsqueeze(2).repeat(batch,1,1,1) # [B, N, 1, 2]
        '''
        self.grid = (self.uvs[None,:,:]*2-1).unsqueeze(2).repeat(batch,1,1,1) # [B, N, 1, 2]
        self.loc_max = 1
        self.loc_min = -1

        #('=== image', torch.min(image), torch.max(image))

        texture_uvs = torch.tensor(self.uvs, dtype=torch.float32, device=device)
        texture_uvs[:,1] = 1 - texture_uvs[:,1]
        self.texture_uvs = texture_uvs
        #self.tex = TexturesUV(verts_uvs=[texture_uvs], faces_uvs=[self.uvfaces], maps=image.permute(0,2,3,1))

        print('=== ' + template[:-5] + '_segment.png')
        #seg = cv2.imread(template[:-5] + '_segment.png')/255.0
        seg = cv2.imread(template[:-4] + '.jpg')/255.0
        #image = cv2.imread('mask.png')/255.0
        seg = torch.from_numpy(seg[:,:,[2,1,0]].astype('float32'))
        seg = seg.permute(2,0,1).unsqueeze(0).to(device)
        seg = (seg - 0.5) / 0.5

        
        self.seg = TexturesUV(verts_uvs=[texture_uvs], faces_uvs=[self.uvfaces], maps=seg.permute(0,2,3,1))

        self.batch = batch

    def toMesh(self, image, tex=None, poseParam=None): 

        # UV location map to Mesh
        # image: [B, C, H, W]
        # 


        image = (image + 1) / 2
        

        #verts = F.grid_sample(image, grid=self.grid, mode='bilinear')
        verts = grid_sample_gradfix.grid_sample(image, self.grid)
        
        verts = verts[:,:,:,0].transpose(1,2)

        verts = (self.loc_max-self.loc_min) * verts + (self.loc_min)
        verts *= 1.0 

        '''
        print('=== :',template[:-4] + '.jpg')
        image = cv2.imread(template[:-4] + '.jpg')/255.0
        image = torch.from_numpy(image[:,:,[2,1,0]].astype('float32'))
        image = image.permute(2,0,1).unsqueeze(0).to(device)
        image = (image - 0.5) / 0.5
        '''
        if tex is None:
            tex = image 

        tex = tex.permute(0,2,3,1)
               
        meshes = []
        for i in range(self.batch):
            verts_sample = verts[i,:,:]

            texture = TexturesUV(verts_uvs=[self.texture_uvs], faces_uvs=[self.uvfaces], maps=tex[i,:,:,:].unsqueeze(0))   
            mesh = Meshes(verts=[verts_sample], faces=[self.faces], textures=texture)

            meshes.append(mesh)

        seges = []
        for i in range(self.batch):
            verts_sample = verts[i,:,:]       
            mesh = Meshes(verts=[verts_sample], faces=[self.faces], textures=self.seg)

            seges.append(mesh)

        meshes = join_meshes_as_batch(meshes)
        seges = join_meshes_as_batch(seges)
        return meshes, seges

class renderer:# Create a renderer
    
    def __init__(self, device):# Create a renderer

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 
        
        raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=0, 
            faces_per_pixel=50, 
        )

        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=256, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, # soft raster, could be a little bit stable in opt
            faces_per_pixel=50, 
        )


        # Render Editing

        # Initialize a camera.
        # Rotate the object by increasing the elevation and azimuth angles
        R, T = look_at_view_transform(dist=3.5, elev=0, azim=0)
        self.cameras = FoVPerspectiveCameras(fov=40.0, device=device, R=R, T=T)

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. # Move the light location so the light is shining on the face.
        #self.lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        self.lights = DirectionalLights(ambient_color=((0.5, 0.5, 0.5), ),
         diffuse_color=((0.40, 0.40, 0.40), ), 
         specular_color=((0.10, 0.10, 0.10), ), 
         direction=((0.0, 0.0, 1.0), ),device=device)
        #self.lights = AmbientLights(device=device)
        #self.lights.location = torch.tensor([[-2.0, 2.0, -2.0]], device=device)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model

        # for color rendering, we could not use raster_settings_soft, but for now we don't know why, maybe it's related to whether the mesh is closed( compared to demo cow)
        '''
        self.render = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=self.cameras,
                lights=self.lights
            )
        ) 
        '''
        self.render = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=self.cameras,
                lights=self.lights
            )
        )


        # Silhouette renderer 
        self.render_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings_soft
            ),
            shader=SoftSilhouetteShader()
        )

        # Change specular color to green and change material shininess 
        self.materials = Materials(
            device=device,
            specular_color=[[0.4, 0.3, 0.3]],
            shininess=10.0
        )
    
    def run(self, targets):
        return self.render(targets, lights=self.lights, cameras=self.cameras)#,materials=self.materials,)

    def run_silhouette(self, targets):
        return self.render_silhouette(targets, lights=self.lights, cameras=self.cameras)

def uvRender(uvlocmap, render, uvset, render_mask=False, render_seg=False, texture=None, poseParam=None):   
    
    #print('==== uvlocmap:', torch.min(uvlocmap), torch.max(uvlocmap))
    mesh, seg = uvset.toMesh(uvlocmap, texture, poseParam)
    silhouette = None
    segments = None
    
    #vertex_normals = mesh.verts_normals_packed()
    #print(vertex_normals)
    images = render.run(mesh)

    images = images[:,:,:,:3] # drop alpha channel
    images = images.permute(0,3,1,2)

    if render_mask:
        silhouette = render.run_silhouette(mesh)
       
        silhouette = silhouette[:,:,:, 3].unsqueeze(3).repeat(1,1,1,3)
        silhouette = silhouette.permute(0,3,1,2)
    if render_seg:
        segments = render.run(seg)
        #print('=== segments:', segments)
        segments = segments[:,:,:,:3] # drop alpha channel
        segments = segments.permute(0,3,1,2)

    return images, silhouette, segments, mesh