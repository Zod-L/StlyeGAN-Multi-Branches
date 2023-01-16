# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""
import sys
import copy
import os
from time import perf_counter

import cv2
import click
import imageio
import numpy as np
from PIL import Image
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision import transforms, utils

from torchvision.transforms import (Compose, Resize, RandomHorizontalFlip, 
                                    ToTensor, Normalize)

import dnnlib
import legacy
from lpips import LPIPS
from torch_utils.ops import grid_sample_gradfix

from Renderer import renderer, uvOpt, uvRender
from Renderer_DECA import SRenderY, set_rasterizer
import Renderer_DECA_util as util


### uv operations ###
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

def curve_draw(loc, grid1, grid2):
    #loc1 = grid_sample_gradfix.grid_sample(loc, grid=self.grid1.repeat(loc.shape[0],1,1,1), mode='bilinear').squeeze()
    if len(loc.shape)>3:
        grid1.repeat(loc.shape[0],1,1,1)
        grid2.repeat(loc.shape[0],1,1,1)

    loc1 = grid_sample_gradfix.grid_sample(loc, grid=grid1).squeeze()
    loc2 = grid_sample_gradfix.grid_sample(loc, grid=grid2).squeeze()
    distmap = loc2-loc1
    return distmap

def project(
    G,
    w_avg,
    path,
    name,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.001,
    initial_noise_factor       = 0.00,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    weight_lpips_render        = 1e0,
    weight_l2_uvmap            = 1e1,
    weight_curveloss           = 0,
    verbose                    = False,
    mask                       = None,
    uvset                      = None,
    rend                       = None,
    teximg                     = None,
    locimg                     = None,
    render                     = None,
    pinCam                     = None,
    grid                       = None,    
    img_size                   = None,
    device: torch.device
):
    #torch.manual_seed(10)
    print('========', target.shape)
    #assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)


    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    #feature_curve_idx = np.load('facescape/feature_curve_idx.npy')
    feature_curve_idx = np.load('facescape/feature_points_idx.npy')
    gird_feature_curve = grid[:,feature_curve_idx,:, :]        

    target_feature_curve = F.grid_sample(locimg, gird_feature_curve, align_corners=False)


    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target


    if target_images.shape[2] > 256:
        target_images1 = F.interpolate(target_images, size=(256, 256), mode='area')

    w_init = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=False) # pylint: disable=not-callable

    w_opt = torch.tensor(w_avg.repeat(G.mapping.num_ws, axis=1), dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    lpips_loss = LPIPS(net='vgg', verbose=False).to(device).eval()
    l2use = torch.nn.MSELoss()

    synbuf = None
    best_score = 10.0

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        #w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.

        ws = w_opt#.repeat([1, G.mapping.num_ws, 1])

        
        synth_loc = G.synthesis(ws, noise_mode='none')
        synth_loc = synth_loc[:,:3,:,:]


        pred_verts = F.grid_sample(synth_loc, grid, align_corners=False)
 
        pred_verts = pred_verts.squeeze(3).permute((0,2,1))
        pred_trans_verts = util.batch_orth_proj(pred_verts, pinCam)
        pred_trans_verts[:,:,1:] = -pred_trans_verts[:,:,1:]
        pred_ops = render(pred_verts, pred_trans_verts, teximg, h=img_size, w=img_size, background=None)

        syn_imgs = pred_ops['normal_images'] #(-1.0, 1.0)
        syn_mask = pred_ops['alpha_images'] #(0.0, 1.0)   
        syn_normal=pred_ops['normal_images']       

        syn_feature_curve = F.grid_sample(synth_loc, gird_feature_curve, align_corners=False)
        
        #exit()
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        #syn_imgs = (syn_imgs + 1) * (255/2)# * syn_mask #(0.0, 255.0)

        ### loss cal 
        maskloss = 0.0

        if mask is not None:
            maskloss = F.mse_loss(mask, syn_mask)
        

        l2loss = F.mse_loss(target_images, syn_imgs)
        l2loss2 = F.mse_loss(locimg, synth_loc)
        l2loss_feature_curve = F.mse_loss(target_feature_curve[:,:2,:,:], syn_feature_curve[:,:2,:,:])


        if syn_imgs.shape[2] > 256:
            synth_images1 = F.interpolate(syn_imgs, size=(256, 256), mode='area')

        loss_lpips = lpips_loss(synth_images1, target_images1).squeeze()

    

        styleloss = F.mse_loss(ws, w_init)
        weight_l2_render, weight_l2_mask, weight_l2_curve = 4, 1, 1000
        loss = weight_lpips_render * loss_lpips +  weight_l2_render * l2loss + 0.5 * l2loss + weight_l2_mask * maskloss + 1e0 * styleloss + weight_l2_curve*l2loss_feature_curve\
             + weight_l2_uvmap * l2loss2

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            logprint(f'step {step+1:>4d}/{num_steps}: lpips {loss_lpips:<4.4f} ls {l2loss:<4.4f} lsuv {weight_l2_curve*l2loss_feature_curve:<4.4f} mask {maskloss:<4.4f} loss {float(loss):<5.5f} lr {float(lr)::<5.5f}')
            result_path = f'{path}/L{weight_l2_render}M{weight_l2_mask}R{weight_lpips_render}C{weight_l2_curve}'
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            utils.save_image(syn_imgs, f'{result_path}/{name}_output.png',
                            nrow=int(2), normalize=True, range=(-1, 1),)


        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

    return w_out[-1]

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=False, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--w_curve',                   help='weight for curve loss', type=float, default=1e2, show_default=True)
@click.option('--w_render',                   help='weight for render loss', type=float, default=1e-1, show_default=True)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    w_curve: float,
    w_render: float,
):
    """Project given image to the latent space of pretrained network pickle.
    Examples:
    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda', 4)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    

    mask_pil = cv2.imread('./facescape/1_1_neutral.png')
    mask_pil = cv2.resize(mask_pil,(512,512))
    cv2.imwrite('facescape/test.png', mask_pil)
    mask = np.zeros([512,512,3])
    ind = np.tile(np.sum(mask_pil,axis=2)[:,:,np.newaxis], (1,1,3))
    cv2.imwrite('facescape/test1.png', ind)
    mask[ind>50] = 1.0
    cv2.imwrite(f'{outdir}/mask.png', mask*255)
    mask = torch.from_numpy(mask).to(device)
    mask = mask.permute(2,0,1)

    ### renderer
    uvset = uvOpt('facescape/1_neutral.obj', device, 1)
    rend = renderer(device)

    template = './facescape/2_neutral.obj'  
    #emplate = './facescape/meanShape_vt_simp_render.obj'  
    set_rasterizer('pytorch3d')
    img_size = 512
    batch_size=1
    from local2mesh.mesh import load_obj_mesh
    _, _, uvs, _ = load_obj_mesh(template,with_texture=True)

    #uvs = uvs * 1.4
    #uvs[:,1] = 1 - uvs[:,1]
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    #print('\n === debug: uvs---', uvs.shape)
    grid = (uvs[None,:,:]*2-1).unsqueeze(2).repeat(batch_size,1,1,1) # [B, N, 1, 2]
    
    #grid = (uvs[None,:,:]*2-1).repeat(batch_size,1,1) # [B, N, 2]    

    render = SRenderY(img_size, obj_filename=template, uv_size=512, rasterizer_type='pytorch3d').to(device)
    pinCam = torch.FloatTensor([0.8, 0.0, 0.0]).unsqueeze(0).repeat(batch_size,1).to(device)

    
    # Compute w stats.
    w_avg_samples=50000
    #logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    np.random.seed(123)
    z_samples = np.random.randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    if not target_fname[-4:] == '.npy':
        path = target_fname
        target_fname = os.listdir(path)
        target = []
        for i in range(len(target_fname)):
            
            if target_fname[i][-4:] == '.npy':
                target.append(path + target_fname[i])


        for target_fname in target:
            name = target_fname.split('/')[-1][:-4]
            print('=== ', target_fname)
        # load target locmap.

            img = np.load(target_fname).astype(np.float32)

            if np.max(img)>10:
                target_uint8 = img.astype(np.uint8)
                img = img/255.0

            img = (img - 0.5) / 0.5 # [-1,1]
            locimg = torch.from_numpy(img).to(device)

            locimg = locimg.permute(2,0,1)
        
            #imgs1.append(img)
            #break
            if len(locimg.shape)<4:
                locimg = locimg.unsqueeze(0)

            with torch.no_grad():
        #        imgs, mask, segments = uvRender(locimg, rend, uvset, render_mask=True, texture=teximg)
                gt_verts = F.grid_sample(locimg, grid, align_corners=False)
                #print('\n === debug: pred_verts---', pred_verts.shape)
                gt_verts = gt_verts.squeeze(3).permute((0,2,1))
                gt_trans_verts = util.batch_orth_proj(gt_verts, pinCam)
                gt_trans_verts[:,:,1:] = -gt_trans_verts[:,:,1:]
                gt_ops = render(gt_verts, gt_trans_verts, locimg, h=img_size, w=img_size, background=None)

                imgs = gt_ops['normal_images']
                mask = gt_ops['alpha_images'] 
                normal=gt_ops['normal_images']           

            if not os.path.exists(f"{path}/input"):
                os.makedirs(f"{path}/input")
            utils.save_image(imgs, f'{path}/input/{name}_input.png',
                                nrow=int(2), normalize=True, range=(-1, 1),)
            utils.save_image(locimg, f'{path}/input/{name}_input_loc.png',
                                nrow=int(2), normalize=True, range=(-1, 1),)
                            
        #imgs1 = torch.stack(imgs1, 0).to(device)
        #print(imgs1.shape)

        # Optimize projection.
            start_time = perf_counter()
            projected_w_steps = project(
                G,
                w_avg,
                path,
                name,
                target=imgs, # pylint: disable=not-callable
                teximg=locimg,
                locimg=locimg,
                num_steps=num_steps,
                device=device,
                verbose=True,
                mask=mask,
                uvset=uvset,
                rend=rend,
                render=render,
                pinCam=pinCam,
                grid=grid,
                img_size=img_size,
                weight_curveloss=w_curve,
                weight_lpips_render=w_render
            )
            print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            np.save(f'{outdir}/{name}.npy', projected_w_steps.unsqueeze(0).cpu().numpy())
        exit()
    exit()
    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='none')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil = PIL.Image.fromarray(target_uint8, 'RGB')
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='none')
    np.save(f'{outdir}/{name}_proj.npy', synth_image.squeeze().permute(1,2,0).cpu().numpy())
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------