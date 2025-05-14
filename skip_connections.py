import torch
from torch import nn

def skip_connection(
    in_features, out_features, n_dim = 2, bias = False, skip_type = "soft-gating"
):
    if skip_type.lower() == "soft-gating":
        return SoftGating1(
            in_features = in_features,
            out_features = out_features,
            bias = bias,
            n_dim = n_dim,
        )
    elif skip_type.lower() == "linear":
        return Flattened1dConv(in_channels = in_features,
                               out_channels = out_features,
                               kernel_size = 1,
                               bias = bias,)
    elif skip_type.lower() == "identity":
        return nn.Identity()

class SoftGating1(nn.Module):
    def __init__(self, in_features, out_features = None, n_dim = 1, bias = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size = 3, padding = 1, groups = in_features)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid(),
        )
        self.spatial_mlp = nn.Conv1d(in_features, 1, kernel_size = 1)
    def forward(self, x):
        spatial_fea = self.conv1(x)
        channel_fea = self.channel_mlp(spatial_fea.mean(dim = -1)).unsqueeze(2) # (B, C, 1)
        spatial_weight = torch.sigmoid(self.spatial_mlp(spatial_fea)) # (B, 1, L)
        x = x * channel_fea * spatial_weight
        return x

class Flattened1dConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias=False):

        super().__init__()
        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              bias = bias)
    def forward(self, x):
        size = list(x.shape)
        x = x.view(*size[:2], -1)
        x = self.conv(x)
        x = x.view(size[0], self.conv.out_channels, *size[2:])
        return x


