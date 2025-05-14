import torch
from torch import nn
from spectralconv import SpectralConv
import torch.nn.functional as F
from skip_connections import skip_connection
from channel_mlp import ChannelMLP
from differential_conv import FiniteDifferenceConvolution

class FNOBlocks(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            n_layers = 4,
            max_n_modes = None,
            channel_mlp_dropout = 0,
            channel_mlp_expansion = 0.5,
            non_linearity = F.relu,
            norm = None,
            fno_skip = "linear",
            channel_mlp_skip = "soft-gating",
            separable = False,
            conv_module = SpectralConv,
            seq_size = 1200,
            fin_diff_kernel_size = 3,
            conv_padding_mode = 'reflect',
            diff_layers = True,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.max_n_modes = max_n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip

        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.separable = separable
        self.non_linearity = non_linearity
        self.seq_size = seq_size

        self.fin_diff_kernel_size = fin_diff_kernel_size
        self.conv_padding_mode = conv_padding_mode
        if isinstance(diff_layers, bool):
            diff_layers = [diff_layers] * n_layers
        self.diff_layers = diff_layers

        self.convs = nn.ModuleList([
                conv_module(
                self.in_channels,
                self.out_channels,
                self.n_modes,
                bias = True,
                separable = separable,
                max_n_modes = max_n_modes,
                init_std = 'auto',
                fft_norm = 'forward',
            ) 
            for _ in range(n_layers)])
        
        self.diff_groups = 1
        self.differential = nn.ModuleList(
            [
                FiniteDifferenceConvolution(self.in_channels, self.out_channels,
                                            self.n_dim, self.fin_diff_kernel_size, 
                                            self.diff_groups, self.conv_padding_mode)
                for _ in range(sum(self.diff_layers))
            ]
        )
        
        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type = fno_skip,
                    n_dim = self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.channel_mlp_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type = channel_mlp_skip,
                    n_dim = self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.n_norms = 1
        if norm is None:
            self.norm = None
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups = 1, num_channels = self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "batch_norm":
            self.norm = nn.ModuleList(
                [
                    nn.BatchNorm1d(num_features = self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "layer_norm":
            self.norm = nn.ModuleList(
                [
                    nn.LayerNorm(normalized_shape=(self.out_channels, self.seq_size))
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

    def forward(self, x, index = 0):
        return self.forward_with_postactivation(x, index)
    
    def forward_with_postactivation(self, x, index = 0):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_fno = self.convs[index](x)
        x_diff = self.differential[index](x, 1200)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno + x_diff + x_skip_channel_mlp

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)
    
class SubModule(nn.Module):

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)













