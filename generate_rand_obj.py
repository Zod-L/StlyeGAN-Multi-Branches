import torch
import legacy
import dnnlib
from global2mesh.uv2mesh_withTex_liyi import global2mesh
from em2mesh.uv2mesh_stretch_yi import em2mesh
import os
from tqdm import tqdm
device = torch.device('cuda', 0)
z = torch.randn(16, 256).to(device)
c = (torch.LongTensor([3]).repeat(16))
c = torch.nn.functional.one_hot(c, num_classes=4).to(device)


with dnnlib.util.open_url("out/emen-feat-quadruple/fuse/00000-stylegan2-raw-gpus8-batch32-gamma10/network-snapshot-002800.pkl") as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
face = (G(z, c, noise_mode='none') + 1) * 127.5
face = face.cpu()
os.makedirs("./experiment/random/", exist_ok=True)

for i in tqdm(range(16)):
        global2mesh(f"experiment/random/triple-{i}", torch.full(face[i, :3, :, :].shape, 128), face[i, :3, :, :])
        #em2mesh(f"experiment/random/local-{i}", torch.full(face[i, 3:, :, :].shape, 256), face[i, 3:6, :, :])

del G
with dnnlib.util.open_url("out/em-triple/fuse/00000-stylegan2-fuse-gpus8-batch64-gamma10/network-snapshot-006048.pkl") as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

        
face = (G(z, c, noise_mode='none') + 1) * 127.5
face = face.cpu()
for i in tqdm(range(16)):
        global2mesh(f"experiment/random/double-{i}", torch.full(face[i, :3, :, :].shape, 128), face[i, :3, :, :])