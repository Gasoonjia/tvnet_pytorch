import os
import numpy as np
from tvnet import TVNet
from frame_dataset import frame_dataset
from train_options import arguments
import torch.utils.data as data
from tvnet import TVNet
import scipy.io as sio


if __name__ == '__main__':
    args = arguments.parse()
    dataloader = data.Dataloader(frame_dataset(args))
    model = TVNet()

    for i, data in enumerate(dataloader):
        u1, u2, rho = model(data[0], data[1])
        u1_np = np.squeeze(u1.to_numpy())
        u2_np = np.squeeze(u2.to_numpy())
        u1_np = np.squeeze(u1_np)
        u2_np = np.squeeze(u2_np)
        flow_mat = np.zeros([h, w, 2])
        flow_mat[:, :, 0] = u1_np
        flow_mat[:, :, 1] = u2_np

        if not os.path.exists('result'):
            os.mkdir('result')
        res_path = os.path.join('result', 'result.mat')
        sio.savemat(res_path, {'flow': flow_mat})