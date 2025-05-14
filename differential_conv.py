import torch
import torch.nn as nn
import torch.nn.functional as F

class FiniteDifferenceConvolution(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            n_dim, 
            kernel_size = 3, 
            groups = 1, 
            padding = 'reflect'):
        super().__init__()
        conv_module = getattr(nn, f"Conv{n_dim}d")
        self.conv_function = getattr(F, f"conv{n_dim}d")
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.groups = groups
        self.n_dim = n_dim

        if padding == 'periodic':
            self.padding_mode = 'circular'
        elif padding == 'replicate':
            self.padding_mode = 'replicate'
        elif padding == 'reflect':
            self.padding_mode = 'reflect'
        elif padding == 'zeros':
            self.padding_mode = 'zeros'

        self.pad_size = kernel_size // 2
        self.conv = conv_module(in_channels, out_channels, kernel_size = kernel_size, 
                                padding = 'same', padding_mode = self.padding_mode,
                                bias = False, groups = groups)
        self.weight = self.conv.weight

    def forward(self, x, grid_width):
        conv = self.conv(x)
        conv_sum = torch.sum(self.conv.weight, dim = tuple([i for i in range(2, 2 + self.n_dim)]), keepdim = True)
        conv_sum = self.conv_function(x, conv_sum, groups = self.groups)
        return (conv - conv_sum) / grid_width


























