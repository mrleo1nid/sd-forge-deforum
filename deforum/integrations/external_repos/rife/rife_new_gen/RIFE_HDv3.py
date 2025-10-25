import os, sys
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from ..model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet_HDv3 import *
import torch.nn.functional as F
from ..model.loss import *
sys.path.append('../../')
from deforum.utils.general import checksum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 3.9
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)
         
    def load_model(self, path, rank, deforum_models_path):
        
        download_rife_model(path, deforum_models_path)

        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(os.path.join(deforum_models_path,'{}.pkl').format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load(os.path.join(deforum_models_path,'{}.pkl').format(path), map_location ='cpu')), False)

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3]
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[3] - gt).abs().mean()
        loss_smooth = self.sobel(flow[3], flow[3]*0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[3], {
            'mask': mask,
            'flow': flow[3][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }

def download_rife_model(path, deforum_models_path):
    # RIFE v4.6 is stable and widely compatible
    # Google Drive file ID: 1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_
    import hashlib

    options = {'RIFE46': (
               '1da8ec98395d7e8e08e5e13e9a5b92da173159abbeca5c47d020edcd1d3cb0b408fa4cc0005141d57d8d526960a5e1ec8bf4948a73eadbea15b6f289574394a2',
               '1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_')}
    if path in options:
        target_file = f"{path}.pkl"
        target_path = os.path.join(deforum_models_path, target_file)
        if not os.path.exists(target_path):
            import gdown
            print(f"Downloading RIFE model {path} from Google Drive...")

            # RIFE v4.6 is distributed as a direct .pkl file (not a ZIP archive)
            gdown.download(id=options[path][1], output=target_path, quiet=False)

            # Verify checksum of downloaded file
            if checksum(target_path, hashlib.sha512) != options[path][0]:
                os.remove(target_path)
                raise Exception(f"Checksum mismatch for {target_file}. Please download manually from: https://drive.google.com/file/d/{options[path][1]}/view and place in: " + deforum_models_path)

            print(f"Successfully downloaded RIFE model to {target_path}")
