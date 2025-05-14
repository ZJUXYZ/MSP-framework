import torch
import torch.nn as nn
import torch.nn.functional as F
from fno_block import FNOBlocks
from spectralconv import SpectralConv

class FNO(nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            hidden_channels,
            n_modes,
            n_layers = 4,
            projection_channel_ratio = 2,
            non_linearity = F.relu,
            channel_mlp_dropout = 0,
            channel_mlp_expansion = 0.5,
            norm = None,
            fno_skip = "linear",
            channel_mlp_skip = "soft-gating",
            max_n_modes = None,
            separable = False,
            conv_module = SpectralConv,
            fin_diff_kernel_size = 3,
            conv_padding_mode = 'reflect',
            diff_layers = True,
            **kwargs
    ):
        super().__init__()
        self.n_dim = len(n_modes)
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        self.non_linearity = non_linearity
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.separable = separable

        self.fno_blocks = FNOBlocks(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            n_modes = self.n_modes,
            n_layers = n_layers,
            max_n_modes = max_n_modes,
            channel_mlp_dropout = channel_mlp_dropout,
            channel_mlp_expansion = channel_mlp_expansion,
            non_linearity = non_linearity,
            norm = norm,
            fno_skip = fno_skip,
            channel_mlp_skip = channel_mlp_skip,
            separable = separable,
            conv_module = conv_module,
            fin_diff_kernel_size = fin_diff_kernel_size,
            conv_padding_mode = conv_padding_mode,
            diff_layers = diff_layers,
            **kwargs
        )

        lifting_in_channels = self.in_channels

        self.lifting = nn.Linear(lifting_in_channels, self.hidden_channels)
        self.projection1 = nn.Linear(self.hidden_channels, self.projection_channels)
        self.projection2 = nn.Linear(self.projection_channels, out_channels)
        self.conv1 = nn.Conv1d(8, hidden_channels, kernel_size = 1)

    def forward(self, x1, x2, **kwargs):
        x1 = self.conv1(x1)
        x2 = self.lifting(x2.permute(0, 2, 1)).permute(0, 2, 1)
        x = x1 + x2

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)

        x = self.projection1(x.permute(0, 2, 1))
        x = self.non_linearity(x)
        x = self.projection2(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

















