import torch
import torch.nn as nn
import numpy as np

def get_module_list(module, n_modules):
    ml = nn.ModuleList()
    for _ in range(n_modules):
        ml.append(module())
    return ml

def conv2d_padding_same(input_size, input_channels, output_channels, kernel_size, \
                        stride=[1,1], bias=True, weight=None, padding_value=0):
    """
        This function is following tensorflow padding style, indicating that it tries to
        pad evenly left(top) and right(bottom), but if the amount of columns to be added is odd, 
        it will add the extra column to the right(bottom). 
    """
    if weight is not None:
        weight = np.asarray(weight)

    assert weight is None or list(weight.shape) == [output_channels, input_channels, *kernel_size]

    height = float(input_size[2])
    width  = float(input_size[3])

    out_size = np.ceil([height / stride[0], width / stride[1]])
    padding_vertical = (out_size[0] - 1) * stride[0] + kernel_size[0] - height
    padding_horizontal = (out_size[1] - 1) * stride[1] + kernel_size[1] - width

    padding_left = int(np.floor(padding_horizontal / 2))
    padding_right = int(np.ceil(padding_horizontal / 2))
    padding_top = int(np.floor(padding_vertical / 2))
    padding_bottom = int(np.ceil(padding_vertical / 2))

    assert padding_left + padding_right == padding_horizontal, "{}, {}, {}".format(padding_left, padding_right, padding_horizontal)
    
    padding_layer = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), padding_value)
    conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, bias=bias)
    if weight is not None:
        conv_layer.weight.data = torch.DoubleTensor(weight)
    
    return nn.Sequential(padding_layer, conv_layer)