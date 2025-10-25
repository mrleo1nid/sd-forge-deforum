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
    # RIFE v4.25 is the recommended default version for most scenes
    # Google Drive file ID: 1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg
    import hashlib
    import zipfile
    import shutil

    options = {'RIFE425': (
               'bae7f128eaecffc9cc146ce198e891770b9008b5f1071c87ab6938279dd3293f90f484921e29564d4fbf3b8e41db56c57fe49d091c8260fb11c6de9e38907543',
               '1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg')}
    if path in options:
        target_file = f"{path}.pkl"
        target_path = os.path.join(deforum_models_path, target_file)
        if not os.path.exists(target_path):
            import gdown
            print(f"Downloading RIFE model {path} from Google Drive...")

            # Download to temporary file
            temp_zip_path = os.path.join(deforum_models_path, f"{path}_temp.zip")
            gdown.download(id=options[path][1], output=temp_zip_path, quiet=False)

            # Verify checksum of downloaded ZIP
            if checksum(temp_zip_path, hashlib.sha512) != options[path][0]:
                os.remove(temp_zip_path)
                raise Exception(f"Checksum mismatch for {target_file}. Please download manually from: https://drive.google.com/file/d/{options[path][1]}/view and place in: " + deforum_models_path)

            # Extract flownet.pkl from train_log/ directory
            print(f"Extracting {path} model from archive...")
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    # Extract train_log/flownet.pkl to target location
                    with zip_ref.open('train_log/flownet.pkl') as source:
                        with open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                print(f"Successfully extracted RIFE model to {target_path}")
            except KeyError:
                os.remove(temp_zip_path)
                raise Exception(f"Archive structure unexpected - train_log/flownet.pkl not found in downloaded file")
            finally:
                # Clean up temporary ZIP file
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
