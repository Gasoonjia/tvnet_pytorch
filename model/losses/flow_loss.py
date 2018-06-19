import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from utils import *

class flow_loss(nn.Module):
    def __init__(self, args):
        super(flow_loss, self).__init__()

        self.lbda = args.lbda
        self.data_size = args.data_size
        self.gradient_kernels = get_module_list(self.get_gradient_kernel, 2)

    
    def get_gradient_kernel(self):
        gradient_block = nn.ModuleList()
        input_size = (1, 1, 480, 480) # NOTE：should change it afterwards. 

        conv_x = conv2d_padding_same(input_size, 1, 1, [1, 2], bias=False, weight=[[[[-1, 1]]]])
        gradient_block.append(conv_x)

        conv_y = conv2d_padding_same(input_size, 1, 1, [2, 1], bias=False, weight=[[[[-1], [1]]]])
        gradient_block.append(conv_y)

        return gradient_block
    
    def warp_image(self, x, u, v):
        assert len(x.size()) == 4
        assert len(u.size()) == 3
        assert len(v.size()) == 3
        
        u = u / x.size(3) * 2
        v = v / x.size(2) * 2
        theta = torch.cat((u, v), dim=1).cuda().float()

        theta = theta.transpose(1, 2).contiguous().view(x.size(0), x.size(2), x.size(3), 2)
        theta += Variable(meshgrid(x.size(2), x.size(3), x.size(0))).cuda().float()

        trans_image = F.grid_sample(x, theta, (x.size(2), x.size(3)))

        return trans_image
    
    def forward(self, u1, u2, x1, x2):
        u1x, u1y = self.forward_gradient(u1, 0)
        u2x, u2y = self.forward_gradient(u2, 1)

        u1_flat = u1.view(self.data_size[0], 1, self.data_size[2] * self.data_size[3])
        u2_flat = u2.view(self.data_size[0], 1, self.data_size[2] * self.data_size[3])

        x2_warp = self.warp_image(x2, u1_flat, u2_flat)
        x2_warp = x2_warp.view(self.data_size)
        loss = self.lbda * torch.mean(torch.abs(x2_warp - x1)) + torch.mean(
            torch.abs(u1x) + torch.abs(u1y) + torch.abs(u2x) + torch.abs(u2y))
        return loss
    
    def forward_gradient(self, x, n_kernel):
        assert len(x.size()) == 4
        assert x.size(1) == 1 # grey scale image

        diff_x = self.gradient_kernels[n_kernel][0](x)
        diff_y = self.gradient_kernels[n_kernel][1](x)

        diff_x_valid = diff_x[:, :, :, :-1]
        last_col = Variable(torch.zeros(diff_x_valid.size(0), diff_x_valid.size(1), diff_x_valid.size(2), 1).float().cuda())
        diff_x = torch.cat((diff_x_valid, last_col), dim=3)

        diff_y_valid = diff_y[:, :, :-1, :]
        last_row = Variable(torch.zeros(diff_x_valid.size(0), diff_x_valid.size(1), 1, diff_y_valid.size(3)).float().cuda())
        diff_y = torch.cat((diff_y_valid, last_row), dim=2)

        return diff_x, diff_y