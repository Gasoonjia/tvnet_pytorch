import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import os

def get_module_list(module, n_modules):
    ml = nn.ModuleList()
    for _ in range(n_modules):
        ml.append(module())
    return ml

# def conv2d_padding_same(input_size, input_channels, output_channels, kernel_size, \
#                         stride=[1,1], bias=True, weight=None, padding_value=0):
#     """
#         This function is following tensorflow padding style, indicating that it tries to
#         pad evenly left(top) and right(bottom), but if the amount of columns to be added is odd, 
#         it will add the extra column to the right(bottom). 
#     """
#     if weight is not None:
#         weight = np.asarray(weight)

#     assert weight is None or list(weight.shape) == [output_channels, input_channels, *kernel_size]

#     height = float(input_size[2])
#     width  = float(input_size[3])

#     out_size = np.ceil([height / stride[0], width / stride[1]])
#     padding_vertical = (out_size[0] - 1) * stride[0] + kernel_size[0] - height
#     padding_horizontal = (out_size[1] - 1) * stride[1] + kernel_size[1] - width

#     padding_left = int(np.floor(padding_horizontal / 2))
#     padding_right = int(np.ceil(padding_horizontal / 2))
#     padding_top = int(np.floor(padding_vertical / 2))
#     padding_bottom = int(np.ceil(padding_vertical / 2))

#     assert padding_left + padding_right == padding_horizontal, "{}, {}, {}".format(padding_left, padding_right, padding_horizontal)
    
#     padding_layer = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), padding_value)
#     conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size, stride=tuple(stride), bias=bias)
#     if weight is not None:
#         conv_layer.weight.data = torch.FloatTensor(weight)
    
#     return nn.Sequential(padding_layer, conv_layer)


def im_tensor_to_numpy(x):
    transpose = transforms.ToPILImage()
    x = np.asarray(transpose(x))
    return x


def save_im_tensor(x, addr):
    x = x.detach().cpu().float()
    transpose = transforms.ToPILImage()
    x = transpose(x[0])
    x.save(addr)


def save_flow_to_img(flow, h, w, c, name='result.png'):
    hsv = np.zeros((h, w, c), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 2] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    res_img_path = os.path.join('result', name)
    cv2.imwrite(res_img_path, rgb)

def torch_where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)

def meshgrid(height, width, n_repeat):
    # print(height, width, n_repeat)
    x_t = torch.matmul(torch.ones(height, 1), 
                        torch.transpose(torch.linspace(-1.0, 1.0, width)[:, None], 1, 0))
    # print(x_t)
    y_t = torch.matmul(torch.linspace(-1.0, 1.0, height)[:, None], torch.ones(1, width))
    # print(y_t)

    x_t_flat = x_t.view(1, -1)
    y_t_flat = y_t.view(1, -1)

    grid = torch.cat([x_t_flat, y_t_flat])[None, ...].cuda().view(-1)
    grid = grid.repeat(n_repeat)
    grid = grid.view(n_repeat, 2, -1)
    grid = grid.permute(0, 2, 1).contiguous().view(n_repeat, height, width, 2)

    # print(grid[0, :, :, 0], x_t)

    return grid