import os
import os.path as osp
import sys
sys.path.append('core')

from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import init
from tqdm import tqdm
import random
from core.utils.frame_utils import read_gen

# Define a custom dataset class
class PETDataset(torch.utils.data.Dataset):
    def __init__(self, split='training', root='../FlowFormer-Official/datasets/pet'):
        self.image_list = []
        self.flow_list = []
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, 'clean')

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.v')))
            for i in range(len(image_list)-1):
                self.image_list += [[image_list[i], image_list[i+1]]]

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root,
                                            scene, '*.mvf')))     
        print("Traning data size: ", len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
            index = index % len(self.image_list)
            valid = None
            flow = read_gen(self.flow_list[index])

            img1 = read_gen(self.image_list[index][0])
            img2 = read_gen(self.image_list[index][1])

            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)


            ## add noise to images
           # img1 = img1 + np.random.normal(0, 0.1, img1.shape)
           # img2 = img2 + np.random.normal(0, 0.1, img2.shape)

            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()

            # clamp images to [0,1]
            img1 = torch.clamp(img1, 0,1)
            img2 = torch.clamp(img2, 0,1)

            flow = torch.from_numpy(flow).float() 

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            input_images = torch.stack([img1,img2], dim=0)
            flow = flow.permute(2, 0, 1)
            return input_images, flow, valid.float()