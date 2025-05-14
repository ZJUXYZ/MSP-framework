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


# class SoftGating(nn.Module):

#     def __init__(self, in_features, out_features = None, n_dim = 1, bias = False):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
#         if bias:
#             self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
#         else:
#             self.bias = None

#     def forward(self, x):

#         if self.bias is not None:
#             return self.weight * x + self.bias
#         else:
#             return self.weight * x
        
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
        # x.shape: b, c, x1, ..., xn x_ndim > 1
        size = list(x.shape)
        # flatten everything past 1st data dim
        x = x.view(*size[:2], -1)
        x = self.conv(x)
        # reshape x into an Nd tensor b, c, x1, x2, ...
        x = x.view(size[0], self.conv.out_channels, *size[2:])
        return x

class Flattened1dConv1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias=False):

        super().__init__()
        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              bias = bias)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.context_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_skip = self.conv(x)
        x_gap = self.gap(x_skip).squeeze(-1)
        context = self.context_mlp(x_gap).unsqueeze(-1)
        x_skip = x_skip * context
        return x_skip




















