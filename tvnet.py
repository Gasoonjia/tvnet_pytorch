import numpy as np
import torch 
import torch.nn as nn
import torch.autograd.variable as Variable
import torch.nn.functional as F
import torchvision as tv

GRAD_IS_ZERO = 1e-12

class TVNet(nn.Module):

    def __init__(self, 
                 img_size,   # size for images being input
                 zfactor=0.5,  # factor for building the image piramid
                 max_scales=5, # maximum number of scales for image piramid
                 ):

        self.zfactor = zfactor

        self.height, self.width = img_size
        n_scales = 1 + np.log(np.sqrt(height ** 2 + width ** 2) / 4.0) / np.log(1 / self.zfactor)
        self.n_scales = min(n_scales, self.max_scales)

        tvnet = nn.ModuleList()

        for ss in range(n_scales):
            tvnet.append(TVNet_Scale())
        
        self.gray_conv = self.get_gray_conv()
        self.gaussian_conv = self.get_gaussian_conv()
        
    
    def get_gray_conv(self):
        gray_conv = nn.Conv2d(3, 1, kernel_size=[1, 1], bias=False, padding=[0, 0])
        gray_conv.weight.data = [[0.114], [0.587], [0.299]]

        return gray_conv

    def get_gaussian_conv(self):
        gaussian_conv = nn.Conv2d(1, 1, kernel_size=[5, 5], bias=False)
        gaussian_conv.weight.data = [[0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
                                     [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                     [0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
                                     [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                     [0.000874, 0.006976, 0.01386, 0.006976, 0.000874]]
        return gaussian_conv
    
    def gray_scale_image(self, x):
        assert len(x.size()) == 4
        assert x.size(1) == 3, 'number of channels must be 3 (i.e. RGB)'

        gray_x = self.gray_conv(x)
        gray_x = torch.floor(gray_x)

        return gray_x
    
    def gaussian_smooth(self, x):
        assert len(x.size()) == 4
        smooth_x = self.gaussian_conv(x)

        return smooth_x
    
    def normalize_images(self, x1, x2):
        min_x1 = x1.min(3).min(2).min(1)
        max_x1 = x1.max(3).max(2).max(1)

        min_x2 = x2.min(3).min(2).min(1)
        max_x2 = x2.max(3).max(2).max(1)

        min_val = torch.min(min_x1, min_x2)
        max_val = torch.max(max_x1, max_x2)

        den = max_val - min_val

        expand_dims = [-1 if i == 0 else 1 for i in range(len(x1.shape))]
        min_val_ex = min_val.reshape(*expand_dims)
        den_ex = den.reshape(*expand_dims)

        x1_norm = torch.where(den > 0, 255. * (x1 - min_val_ex) / den_ex, x1)
        x2_norm = torch.where(den > 0, 255. * (x2 - min_val_ex) / den_ex, x2)

        return x1_norm, x2_norm
    
    def zoom_size(self, height, width, factor):
        new_height = int(float(height) * factor + 0.5)
        new_width = int(float(width) * factor + 0.5)

        return new_height, new_width
        
    def forward(self, x1, x2):
        if x1.size(1) == 3:
            x1 = self.gray_scale_image(x1)
            x2 = self.gray_scale_image(x2)

        norm_imgs = self.normalize_images(grey_x1, grey_x2)

        smooth_x1 = self.gaussian_smooth(norm_imgs[0])
        smooth_x2 = self.gaussian_smooth(norm_imgs[1])

        for ss in range(self.n_scales-1, -1, -1):
            down_sample_factor = self.zfactor * ss
            down_height, down_width = self.zoom_size(self.height, self.width, down_sample_factor)
            
            if ss == n_scales - 1:
                u1 = Variable(tensor.double(torch.zeros(smooth_x1.size(0), down_height, down_width, 1)))
                u2 = Variable(tensor.double(torch.zeros(smooth_x2.size(0), down_height, down_width, 1)))

            down_x1 = self.zoom_image(smooth_x1, down_height, down_width)
            down_x2 = self.zoom_image(smooth_x2, down_height, down_width)

            u1, u2, rho = self.tvnet[ss](down_x1, down_x2, u1, u2)

            if ss == 0:
                return u1, u2, rho
            
            up_sample_factor = zfactor ** (ss - 1)
            up_height, up_width = self.zoom_size(height, width, up_sample_factor)
            u1 = self.zoom_image(u1, up_height, up_width) / zfactor
            u2 = self.zoom_image(u2, up_height, up_width) / zfactor


class TVNet_Scale(nn.Module):

    def __init__(self,
                 tau=0.25,   # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 zfactor=0.5,  # factor for building the image piramid
                #  max_scales=5, # maximum number of scales for image piramid
                 n_iters=5  # maximum number of iterations for optimization
                ):

        self.tau = tau
        self.lbda = lbda
        self.theta = theta
        self.zfactor = zfactor
        self.n_iters = n_iters
        # self.max_scales = max_scales

        self.gradient = nn.ModuleList()
        self.divergence = nn.ModuleList()

        for n_warp in range(warps):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            gradient_warp = nn.ModuleList()
            divergence_warp = nn.ModuleList()
            for n_iter in range(self.n_iters):
                gradient = nn.ModuleList()
                gradient.append(self.get_gradient_block())
                gradient.append(self.get_gradient_block())
                gradient_warp.append(gradient)

                divergence = nn.ModuleList()
                divergence.append(self.get_divergence_block())
                divergence.append(self.get_divergence_block())
                divergence_warp.append(divergence)
            
            self.gradient.append(gradient_warp)
            self.divergence.append(divergence_warp)


    def get_divergence_block(self):
        divergence_block = nn.ModuleList() #[conv_x, conv_y]
        
        conv_x = nn.Conv2d(1, 1, kernel_size=[1, 2], bias=False, padding=[0, 1])
        conv_x.weight.data = [[-1, 1]]
        divergence_block.append(conv_x)

        conv_y = nn.Conv2d(1, 1, kernel_size=[2, 1], bias=False, padding=[1, 0]) #padding = same
        conv_x.weight.data = [[-1], [1]]
        divergence_block.append(conv_x)

        return divergence_block
    

    def get_gradient_block(self):
        gradient_block = nn.ModuleList()

        conv_x = nn.Conv2d(1, 1, kernel_size=[1, 2], bias=False, padding=[0, 1])
        conv_x.weight.data = [[-1, 1]]
        gradient_block.append(conv_x)

        conv_y = nn.Conv2d(1, 1, kernel_size=[2, 1], bias=False, padding=[1, 0]) #padding = same
        conv_x.weight.data = [[-1], [1]]
        gradient_block.append(conv_x)

        return gradient_block


    def forward(self, x1, x2, u1, u2):
        l_t = self.lbda * self.theta
        taut = self.tau / self.theta

        diff2_x, diff2_y = self.centered_gradient(x2, 'x2')

        p11 = torch.zeros_like(x1)
        p12 = torch.zeros_like(x1)
        p21 = torch.zeros_like(x1)
        p22 = torch.zeros_like(x1)

        # p11 = p12 = p21 = p22 = tf.zeros_like(x1) in original tensorflow code, 
        # it seems that each element of p11 to p22 shares a same memory address and I'm not sure if it would make some mistakes or not.

        for n_warp in range(warps):
            u1_flat = u1.reshape(x2.size(0), 1, x2.size(1)*x2.size(2))
            u2_flat = u2.reshape(x2.size(0), 1, x2.size(1)*x2.size(2))

            x2_warp = self.warp_image(x2, u1_flat, u2_flat)
            x2_warp = x2_warp.reshape(x2.size())

            diff2_x_warp = self.warp_image(diff2_x, u1_flat, u2_flat)
            diff2_x_warp = diff2_x_warp.reshape(diff2_x.size())

            diff2_y_warp = self.warp_image(diff2_y, u1_flat, u2_flat)
            diff2_y_warp = diff2_y_warp.reshape(diff2_y.size())

            diff2_x_sq = torch.square(diff2_x_warp)
            diff2_y_sq = torch.square(diff2_y_warp)

            grad = diff2_x_sq + diff2_y_sq + GRAD_IS_ZERO

            rho_c = x2_warp - diff2_x_warp * u1 - diff2_y_warp * u2 - x1

            for n_iter in range(n_iters):
                rho = rho_c + diff2_x_warp * u1 + diff2_y_warp * u2 + GRAD_IS_ZERO

                masks1 = rho < -l_t * grad
                d1_1 = torch.where(masks1, l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
                d2_1 = torch.where(masks1, l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))

                masks2 = rho > l_t * grad
                d1_2 = torch.where(masks2, -l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
                d2_2 = torch.where(masks2, -l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))

                masks3 = (~masks1) & (~masks2) & (grad > self.GRAD_IS_ZERO)
                d1_3 = torch.where(masks3, -rho / grad * diff2_x_warp, torch.zeros_like(diff2_x_warp))
                d2_3 = torch.where(masks3, -rho / grad * diff2_y_warp, torch.zeros_like(diff2_y_warp))

                v1 = d1_1 + d1_2 + d1_3 + u1
                v2 = d2_1 + d2_2 + d2_3 + u2

                u1 = v1 + theta * self.forward_divergence(p11, p12, n_warp, n_iter, 0)
                u2 = v2 + theta * self.forward_divergence(p21, p22, n_warp, n_iter, 1)

                u1x, u1y = self.forward_gradient(u1, n_warp, n_iter, 0)
                u2x, u2y = self.forward_gradient(u2, n_warp, n_iter, 1)

                p11 = (p11 + taut * u1x) / (
                    1.0 + taut * torch.sqrt(torch.square(u1x) + torch.square(u1y) + GRAD_IS_ZERO))
                p12 = (p12 + taut * u1y) / (
                    1.0 + taut * torch.sqrt(torch.square(u1x) + torch.square(u1y) + GRAD_IS_ZERO))
                p21 = (p21 + taut * u2x) / (
                    1.0 + taut * torch.sqrt(torch.square(u2x) + torch.square(u2y) + GRAD_IS_ZERO))
                p22 = (p22 + taut * u2y) / (
                    1.0 + taut * torch.sqrt(torch.square(u2x) + torch.square(u2y) + GRAD_IS_ZERO))

        return u1, u2, rho

    def warp_image(self, x, u, v):
        assert len(x.size()) == 4
        assert len(u.size()) == 3
        assert len(v.size()) == 3
        
        u = u / x.size(3) * 2
        v = v / x.size(2) * 2
        delta = torch.cat((u, v), dim=1)
        
        grid = F.affine_grid(delta, x.size())
        target_img = F.grid_sample(x, grid)

        return target_img
    
    def forward_divergence(self, x, y, n_warp, n_iter, n_block):
        assert len(x.size()) == 4 #[bs, c, h, w]
        assert len(x.size(1)) == 1 # grey scale image

        x_valid = x[:, :, :, :-1]
        first_col = torch.zeros(x.size(0), x.size(1), x.size(2), 1)
        x_pad = torch.cat((first_col, x_valid), dim=3)

        y_valid = y[:, :, :-1, :]
        first_row = torch.zeros(y.size(0), y.size(1), 1, y.size(3))
        y_pad = torch.cat((first_row, y_valid), dim=2)

        diff_x = self.divergence[n_warp][n_iter][n_block][0](x_pad)
        diff_y = self.divergence[n_warp][n_iter][n_block][1](y_pad)

        div = diff_x + diff_y

        return div
    
    def forward_gradient(self, x, n_warp, n_iter, n_block):
        assert len(x.size()) == 4
        assert len(x.size(1)) == 1 # grey scale image

        diff_x = self.gradient[n_warp][n_iter][n_block][0](x)
        diff_y = self.gradient[n_warp][n_iter][n_block][1](x)

        diff_x_valid = diff_x[:, :, :, :-1]
        last_col = torch.zeros(*diff_x_valid.size()[:-1], 1)
        diff_x = torch.cat((diff_x_valid, last_col), dim=3)

        diff_y_valid = diff_y[:, :, :-1, :]
        last_row = torch.zeros(*diff_y_valid.size()[:-2], 1, diff_y_valid.size(3))
        diff_y = torch.cat((diff_y_valid, last_row), dim=2)

        return diff_x, diff_y


