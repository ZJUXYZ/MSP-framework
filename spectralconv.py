import torch
from torch import nn
import tensorly as tl
from tensorly.plugins import use_opt_einsum
import torch.nn.functional as F
from typing import List, Optional, Tuple

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _contract_dense(x, weight, separable = False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order]) # [a, b, c, d]

    weight_syms = list(x_syms[1:])  # [b, c, d]

    if separable:
        out_syms = [x_syms[0]] + list(weight_syms) # [a, b, c, d]
    else:
        weight_syms.insert(1, einsum_symbols[order])  # [b, e, c, d]
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0] #[a, e, c, d]
    
    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight

def get_contract_fun(separable = False):
    if separable:
        return _contract_dense_separable
    else:
        return _contract_dense


class SpectralConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, n_modes, 
            bias = True,
            separable = False,
            max_n_modes = None,
            init_std = "auto",
            fft_norm = "forward",
            ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        else:
            init_std = init_std

        self.fft_norm = fft_norm

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable

        weight_tensor = torch.empty(weight_shape, dtype = torch.cfloat)

        weight_tensor.normal_(0, init_std)
        self.weight = nn.Parameter(weight_tensor)

        # self.weight = nn.Parameter(weight_shape, dtype = torch.cfloat)
        # self.weight.normal_(0, init_std)

        self._contract = get_contract_fun(
            separable = separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
        self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None
    ):
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm = self.fft_norm, dim = fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim = fft_dims[:-1])

        out_dtype = torch.cfloat

        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                              device = x.device, dtype = out_dtype)
        
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]
        if self.separable: 
            slices_w = [slice(None)] # channels
        else:
            slices_w =  [slice(None), slice(None)]
        slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        # print(self.weight.shape)
        weight = self.weight[slices_w]

        if self.separable:
            weight_start_idx = 1
        else:
            weight_start_idx = 2
        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[weight_start_idx:]))]
        slices_x =  [slice(None), slice(None)] # Batch_size, channels

        slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        # print(x[slices_x].shape, weight.shape)

        out_fft[slices_x] = self._contract(x[slices_x], weight, separable = self.separable)

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim = fft_dims[:-1])

        x = torch.fft.irfftn(out_fft, s = mode_sizes, dim = fft_dims, norm = self.fft_norm)
        
        if self.bias is not None:
            x = x + self.bias

        return x
