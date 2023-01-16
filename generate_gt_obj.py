import torch
import click
import legacy
import dnnlib
import os
from global2mesh.uv2mesh_withTex_liyi import generate
from torch_utils import misc
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")




def main():
    fnames = {os.path.relpath(os.path.join(root, fname)) for root, _dirs, files in os.walk("./data") for fname in files}
    fnames = sorted(fname for fname in fnames if (os.path.splitext(fname)[1].lower() == ".npy"))

    for fname in tqdm(fnames):
        face = torch.from_numpy(np.load(fname))
        face = face.permute(2, 0, 1)
        generate(os.path.join("experiment/gt", os.path.basename(fname).split('.')[0]), torch.full(face.shape, 128), face)
    

if __name__ == "__main__":
    main()





