import os
import numpy as np
from tvnet import TVNet
# import data.frame_dataset as frame_dataset
from data.frame_dataset import frame_dataset
from train_options import arguments
import torch.utils.data as data
from tvnet import TVNet
import scipy.io as sio
import cv2
from utils import *


if __name__ == '__main__':
    args = arguments().parse()
    dataset = frame_dataset(args)
    img_size = dataset.img_size
    dataloader = data.DataLoader(dataset)
    model = TVNet(img_size).cuda()

    for i, data in enumerate(dataloader):
        _, c, h, w = data[0].size()
        u1, u2, rho = model(data[0].cuda(), data[1].cuda())
        u1_np = np.squeeze(u1.detach().cpu().numpy())
        u2_np = np.squeeze(u2.detach().cpu().numpy())
        flow_mat = np.zeros([h, w, 2])
        flow_mat[:, :, 0] = u1_np
        flow_mat[:, :, 1] = u2_np

        if not os.path.exists('result'):
            os.mkdir('result')
        res_mat_path = os.path.join('result', 'result.mat')
        sio.savemat(res_mat_path, {'flow': flow_mat})
        if args.visualize:
            save_flow_to_img(flow_mat, h, w, c)