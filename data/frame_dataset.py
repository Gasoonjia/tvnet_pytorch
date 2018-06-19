import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

class frame_dataset(data.Dataset):
    def __init__(self, args):
        self.frame_dir = args.frame_dir
        
        self.frame_addr = np.asarray([os.path.join(self.frame_dir, addr) for addr in os.listdir(self.frame_dir)])
        self.frame_addr.sort()
        self.to_tensor = get_transfrom()
        self.img_size = Image.open(self.frame_addr[0]).convert('RGB').size
    
    def __len__(self):
        return self.frame_addr.shape[0] - 1

    def __getitem__(self, index):
        frame_1 = self.to_tensor(Image.open(self.frame_addr[index]).convert('RGB')).float()
        frame_2 = self.to_tensor(Image.open(self.frame_addr[index+1]).convert('RGB')).float()
        return frame_1, frame_2

def get_transfrom():
    transforms_list = []
    transforms_list = [transforms.ToTensor()]
    return transforms.Compose(transforms_list)