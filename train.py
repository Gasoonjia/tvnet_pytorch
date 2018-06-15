import os
import numpy as np
# import data.frame_dataset as frame_dataset
from data.frame_dataset import frame_dataset
from train_options import arguments
import torch.utils.data as data
from model.network import model
import scipy.io as sio
import cv2
from utils import *


if __name__ == '__main__':
    args = arguments().parse()
    dataset = frame_dataset(args)
    args.data_size = [args.batch_size, 3, *(dataset.img_size)]
    dataloader = data.DataLoader(dataset)
    model = model(args)

    for n_epoch in range(args.n_epoch):
        for i, data in enumerate(dataloader):
            u1, u2 = model.forward(data[0], data[1], need_result=True)
            model.optimize()
        if n_epoch % 100 == 99:
            print(n_epoch + 1)
        elif n_epoch < 100 and (n_epoch + 1) % 10 == 0:
            print(n_epoch + 1)
        elif n_epoch < 10:
            print(n_epoch + 1)

    _, c, h, w = args.data_size
    u1_np = np.squeeze(u1.detach().cpu().data.numpy())
    u2_np = np.squeeze(u2.detach().cpu().data.numpy())
    flow_mat = np.zeros([h, w, 2])
    flow_mat[:, :, 0] = u1_np
    flow_mat[:, :, 1] = u2_np

    if not os.path.exists('result'):
        os.mkdir('result')
    res_mat_path = os.path.join('result', 'result.mat')
    sio.savemat(res_mat_path, {'flow': flow_mat})
    if args.visualize:
        save_flow_to_img(flow_mat, h, w, c)