import numpy as np
import torch 
import torch.nn as nn
import torch.autograd.variable as Variable
import torch.nn.functional as F
from networks.model.spatial_transformer import spatial_transformer
from utils import *

GRAD_IS_ZERO = 1e-12

class TVNet(nn.Module):

    def __init__(self, 
                 img_size,   # size for images being input
                 zfactor=0.5,  # factor for building the image piramid
                 max_scales=5, # maximum number of scales for image piramid
                 ):

        super(TVNet, self).__init__()
        self.zfactor = zfactor
        self.max_scales = max_scales

        self.height, self.width = img_size
        n_scales = 1 + np.log(np.sqrt(self.height ** 2 + self.width ** 2) / 4.0) / np.log(1 / self.zfactor)
        self.n_scales = min(n_scales, self.max_scales)

        self.tvnet_kernels = nn.ModuleList()
        self.zoom_kernels = nn.ModuleList()

        for ss in range(self.n_scales):
            self.zoom_kernels.append(get_module_list(spatial_transformer, 4))
            self.tvnet_kernels.append(TVNet_Scale())

        self.gray_kernels = get_module_list(self.get_gray_conv, 2).train(False)
        self.gaussian_kernels = get_module_list(self.get_gaussian_conv, 2).train(False)
    

    def forward(self, x1, x2):
        if x1.size(1) == 3:
            x1 = self.gray_scale_image(x1, 0)
            x2 = self.gray_scale_image(x2, 1)

        # return x1, x2

        norm_imgs = self.normalize_images(x1, x2)

        smooth_x1 = self.gaussian_smooth(norm_imgs[0], 0)
        smooth_x2 = self.gaussian_smooth(norm_imgs[1], 1)

        for ss in range(self.n_scales-1, -1, -1):
            down_sample_factor = self.zfactor ** ss
            down_height, down_width = self.zoom_size(self.height, self.width, down_sample_factor)
            
            if ss == self.n_scales - 1:
                u1 = Variable(torch.zeros(smooth_x1.size(0), 1, down_height, down_width).double()).cuda()
                u2 = Variable(torch.zeros(smooth_x2.size(0), 1, down_height, down_width).double()).cuda()

            down_x1 = self.zoom_image(smooth_x1, down_height, down_width, ss, 0)
            down_x2 = self.zoom_image(smooth_x2, down_height, down_width, ss, 1)

            u1, u2, rho = self.tvnet_kernels[ss](down_x1, down_x2, u1, u2)

            if ss == 0:
                return u1, u2, rho
            
            up_sample_factor = self.zfactor ** (ss - 1)
            up_height, up_width = self.zoom_size(self.height, self.width, up_sample_factor)
            u1 = self.zoom_image(u1, up_height, up_width, ss, 2) / self.zfactor
            u2 = self.zoom_image(u2, up_height, up_width, ss, 3) / self.zfactor

    
    def get_gray_conv(self):
        input_size = (1, 1, 480, 480) # NOTE：should change it afterwards. 

        gray_conv = conv2d_padding_same(input_size, 3, 1, [1, 1], bias=False, 
                                        weight=[[[[0.114]], [[0.587]], [[0.299]]]])

        return gray_conv


    def get_gaussian_conv(self):
        input_size = (1, 1, 480, 480) # NOTE：should change it afterwards. 

        gaussian_conv = conv2d_padding_same(input_size, 1, 1, [5, 5], bias=False, 
                                            weight=[[[[0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
                                                    [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                                    [0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
                                                    [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                                    [0.000874, 0.006976, 0.01386, 0.006976, 0.000874]]]])
        return gaussian_conv

    
    def gray_scale_image(self, x, n_kernel):
        assert len(x.size()) == 4
        assert x.size(1) == 3, 'number of channels must be 3 (i.e. RGB)'

        gray_x = self.gray_kernels[n_kernel](x)

        return gray_x

    
    def gaussian_smooth(self, x, n_kernel):
        assert len(x.size()) == 4
        smooth_x = self.gaussian_kernels[n_kernel](x)

        return smooth_x

    
    def normalize_images(self, x1, x2):
        min_x1 = x1.min(3)[0].min(2)[0].min(1)[0]
        max_x1 = x1.max(3)[0].max(2)[0].max(1)[0]

        min_x2 = x2.min(3)[0].min(2)[0].min(1)[0]
        max_x2 = x2.max(3)[0].max(2)[0].max(1)[0]

        min_val = torch.min(min_x1, min_x2)
        max_val = torch.max(max_x1, max_x2)

        den = max_val - min_val

        expand_dims = [-1 if i == 0 else 1 for i in range(len(x1.shape))]
        min_val_ex = min_val.view(*expand_dims)
        den_ex = den.view(*expand_dims)

        x1_norm = torch.where(den > 0, 255. * (x1 - min_val_ex) / den_ex, x1)
        x2_norm = torch.where(den > 0, 255. * (x2 - min_val_ex) / den_ex, x2)

        return x1_norm, x2_norm

    
    def zoom_size(self, height, width, factor):
        new_height = int(float(height) * factor + 0.5)
        new_width = int(float(width) * factor + 0.5)

        return new_height, new_width

    
    def zoom_image(self, x, new_height, new_width, n_scale, n_kernel):
        assert len(x.shape) == 4

        theta = torch.zeros(x.size(0), 2, new_height * new_width).cuda()
        zoomed_x = self.zoom_kernels[n_scale][n_kernel](x, theta, (new_height, new_width))
        return zoomed_x.reshape(x.size(0), x.size(1), new_height, new_width)
            


class TVNet_Scale(nn.Module):

    def __init__(self,
                 tau=0.25,   # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 zfactor=0.5,  # factor for building the image piramid
                 warps=5,  # number of warpings per scale
                #  max_scales=5, # maximum number of scales for image piramid
                 n_iters=5  # maximum number of iterations for optimization
                ):

        super(TVNet_Scale, self).__init__()

        self.tau = tau
        self.lbda = lbda
        self.theta = theta
        self.n_warps = warps
        self.zfactor = zfactor
        self.n_iters = n_iters
        # self.max_scales = max_scales

        self.gradient_kernels = nn.ModuleList()
        self.divergence_kernels = nn.ModuleList()
        self.warp_kernels = nn.ModuleList()

        self.centered_gradient_kernels = self.get_centered_gradient_kernel().train(False)

        for n_warp in range(self.n_warps): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.warp_kernels.append(get_module_list(spatial_transformer, 3))

            gradient_warp = nn.ModuleList()
            divergence_warp = nn.ModuleList()
            for n_iter in range(self.n_iters):
                gradient_warp.append(get_module_list(self.get_gradient_kernel, 2))
                divergence_warp.append(get_module_list(self.get_divergence_kernel, 2))
            
            self.gradient_kernels.append(gradient_warp)
            self.divergence_kernels.append(divergence_warp)
    

    def get_gradient_kernel(self):
        gradient_block = nn.ModuleList()
        input_size = (1, 1, 480, 480) # NOTE：should change it afterwards. 

        conv_x = conv2d_padding_same(input_size, 1, 1, [1, 2], bias=False, weight=[[[[-1, 1]]]])
        gradient_block.append(conv_x)

        conv_y = conv2d_padding_same(input_size, 1, 1, [2, 1], bias=False, weight=[[[[-1], [1]]]])
        gradient_block.append(conv_y)

        return gradient_block


    def get_divergence_kernel(self):
        divergence_block = nn.ModuleList() #[conv_x, conv_y]
        input_size = (1, 1, 480, 480) # NOTE：should change it afterwards. 
        
        conv_x = conv2d_padding_same(input_size, 1, 1, [1, 2], bias=False, weight=[[[[-1, 1]]]])
        divergence_block.append(conv_x)

        conv_y = conv2d_padding_same(input_size, 1, 1, [2, 1], bias=False, weight=[[[[-1], [1]]]])
        divergence_block.append(conv_y)

        return divergence_block
    

    def get_centered_gradient_kernel(self):
        centered_gradient_block = nn.ModuleList()
        input_size = (1, 1, 480, 480) # NOTE：should change it afterwards. 

        conv_x = conv2d_padding_same(input_size, 1, 1, [1, 3], bias=False, weight=[[[[-0.5, 0, 0.5]]]])
        centered_gradient_block.append(conv_x)

        conv_y = conv2d_padding_same(input_size, 1, 1, [3, 1], bias=False, weight=[[[[-0.5], [0], [0.5]]]])
        centered_gradient_block.append(conv_y)

        return centered_gradient_block


    def forward(self, x1, x2, u1, u2):
        l_t = self.lbda * self.theta
        taut = self.tau / self.theta

        diff2_x, diff2_y = self.centered_gradient(x2)

        p11 = torch.zeros_like(x1).cuda()
        p12 = torch.zeros_like(x1).cuda()
        p21 = torch.zeros_like(x1).cuda()
        p22 = torch.zeros_like(x1).cuda()

        # p11 = p12 = p21 = p22 = tf.zeros_like(x1) in original tensorflow code, 
        # it seems that each element of p11 to p22 shares a same memory address and I'm not sure if it would make some mistakes or not.

        for n_warp in range(self.n_warps):
            u1_flat = u1.view(x2.size(0), 1, x2.size(2)*x2.size(3))
            u2_flat = u2.view(x2.size(0), 1, x2.size(2)*x2.size(3))

            x2_warp = self.warp_image(x2, u1_flat, u2_flat, n_warp, 0)
            x2_warp = x2_warp.view(x2.size())

            diff2_x_warp = self.warp_image(diff2_x, u1_flat, u2_flat, n_warp, 1)
            diff2_x_warp = diff2_x_warp.view(diff2_x.size())
            # print(diff2_x_warp.size())

            diff2_y_warp = self.warp_image(diff2_y, u1_flat, u2_flat, n_warp, 2)
            diff2_y_warp = diff2_y_warp.view(diff2_y.size())

            diff2_x_sq = diff2_x_warp ** 2
            diff2_y_sq = diff2_y_warp ** 2

            grad = diff2_x_sq + diff2_y_sq + GRAD_IS_ZERO

            rho_c = x2_warp - diff2_x_warp * u1 - diff2_y_warp * u2 - x1

            for n_iter in range(self.n_iters):
                rho = rho_c + diff2_x_warp * u1 + diff2_y_warp * u2 + GRAD_IS_ZERO

                masks1 = rho < -l_t * grad
                d1_1 = torch.where(masks1, l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
                d2_1 = torch.where(masks1, l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))

                masks2 = rho > l_t * grad
                d1_2 = torch.where(masks2, -l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
                d2_2 = torch.where(masks2, -l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))

                masks3 = (~masks1) & (~masks2) & (grad > GRAD_IS_ZERO)
                d1_3 = torch.where(masks3, -rho / grad * diff2_x_warp, torch.zeros_like(diff2_x_warp))
                d2_3 = torch.where(masks3, -rho / grad * diff2_y_warp, torch.zeros_like(diff2_y_warp))

                v1 = d1_1 + d1_2 + d1_3 + u1
                v2 = d2_1 + d2_2 + d2_3 + u2

                u1 = v1 + self.theta * self.forward_divergence(p11, p12, n_warp, n_iter, 0)
                u2 = v2 + self.theta * self.forward_divergence(p21, p22, n_warp, n_iter, 1)

                u1x, u1y = self.forward_gradient(u1, n_warp, n_iter, 0)
                u2x, u2y = self.forward_gradient(u2, n_warp, n_iter, 1)

                p11 = (p11 + taut * u1x) / (
                    1.0 + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + GRAD_IS_ZERO))
                p12 = (p12 + taut * u1y) / (
                    1.0 + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + GRAD_IS_ZERO))
                p21 = (p21 + taut * u2x) / (
                    1.0 + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + GRAD_IS_ZERO))
                p22 = (p22 + taut * u2y) / (
                    1.0 + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + GRAD_IS_ZERO))

        return u1, u2, rho
    
    def centered_gradient(self, x):
        assert len(x.shape) == 4

        diff_x = self.centered_gradient_kernels[0](x)
        diff_y = self.centered_gradient_kernels[1](x)

        # refine the boundary
        first_col = 0.5 * (x[..., 1:2] - x[..., 0:1])
        last_col = 0.5 * (x[..., -1:] - x[..., -2:-1])
        diff_x_valid = diff_x[..., 1:-1]

        diff_x = torch.cat([first_col, diff_x_valid, last_col], dim=-1)
        
        first_row = 0.5 * (x[:, :, 1: 2, :] - x[:, :, 0: 1, :])
        last_row = 0.5 * (x[:, :, -1:, :] - x[:, :, -2:-1, :])
        diff_y_valid = diff_y[:, :, 1:-1, :]
        diff_y = torch.cat([first_row, diff_y_valid, last_row], dim=-2)

        return diff_x, diff_y 

    def warp_image(self, x, u, v, n_warp, n_kernel):
        assert len(x.size()) == 4
        assert len(u.size()) == 3
        assert len(v.size()) == 3
        
        u = u / x.size(3) * 2
        v = v / x.size(2) * 2
        theta = torch.cat((u, v), dim=1)

        trans_image = self.warp_kernels[n_warp][n_kernel](x, theta, (x.size(2), x.size(3)))

        return trans_image
    
    def forward_divergence(self, x, y, n_warp, n_iter, n_kernel):
        assert len(x.size()) == 4 #[bs, c, h, w]
        assert x.size(1) == 1 # grey scale image

        x_valid = x[:, :, :, :-1]
        first_col = torch.zeros(x.size(0), x.size(1), x.size(2), 1).double().cuda()
        x_pad = torch.cat((first_col, x_valid), dim=3)

        y_valid = y[:, :, :-1, :]
        first_row = torch.zeros(y.size(0), y.size(1), 1, y.size(3)).double().cuda()
        y_pad = torch.cat((first_row, y_valid), dim=2)

        diff_x = self.divergence_kernels[n_warp][n_iter][n_kernel][0](x_pad)
        diff_y = self.divergence_kernels[n_warp][n_iter][n_kernel][1](y_pad)

        div = diff_x + diff_y

        return div
    
    def forward_gradient(self, x, n_warp, n_iter, n_kernel):
        assert len(x.size()) == 4
        assert x.size(1) == 1 # grey scale image

        diff_x = self.gradient_kernels[n_warp][n_iter][n_kernel][0](x)
        diff_y = self.gradient_kernels[n_warp][n_iter][n_kernel][1](x)

        diff_x_valid = diff_x[:, :, :, :-1]
        last_col = torch.zeros(diff_x_valid.size(0), diff_x_valid.size(1), diff_x_valid.size(2), 1).double().cuda()
        diff_x = torch.cat((diff_x_valid, last_col), dim=3)

        diff_y_valid = diff_y[:, :, :-1, :]
        last_row = torch.zeros(diff_x_valid.size(0), diff_x_valid.size(1), 1, diff_y_valid.size(3)).double().cuda()
        diff_y = torch.cat((diff_y_valid, last_row), dim=2)

        return diff_x, diff_y


