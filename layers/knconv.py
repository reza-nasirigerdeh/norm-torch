"""
    MIT License
    Copyright (c) 2024 Reza NasiriGerdeh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch
from torch.nn.common_types import _size_2_t, _size_4_t
from typing import Union
from torch import Tensor
from torch.nn import functional as F
from layers.kernel_norm import KernelNorm2d1x1


class KNConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, _size_2_t] = (1, 1),
                 stride: Union[int, _size_2_t] = (1, 1),
                 padding: Union[int, _size_2_t, _size_4_t] = (0, 0),
                 groups: int = 1,
                 bias: bool = True,
                 dropout_p: float = 0.05,
                 eps: float = 1e-5):
        super(KNConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                       dilation=1, groups=groups,
                                       bias=bias, padding_mode='zeros', device=None, dtype=None)

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding, padding) if isinstance(padding, int) else padding
        self.padding = self.padding if len(self.padding) == 4 else (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        self.dropout_p = dropout_p
        self.eps = eps

    def forward(self, input_t: Tensor) -> Tensor:
        # pad input tensor
        x_t = F.pad(input=input_t, pad=self.padding, mode="constant", value=0.0)

        # compute convolution over padded input
        out_conv = F.conv2d(input=x_t, weight=self.weight, bias=None, groups=self.groups, stride=self.stride, padding=0)

        # apply dropout to padded input
        x_t = F.dropout(input=x_t, p=self.dropout_p, inplace=False)

        # unfold padded input (sliding window mechanism)
        x_t = x_t.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        x_t = x_t.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])

        # permute unfolded tensor so that channel,height,width become the last three dimensions
        x_t = x_t.permute(dims=(0, 2, 3, 1, 4, 5))

        # compute mean and variance over unfolded tensor
        var, mu = torch.var_mean(input=x_t, dim=[3, 4, 5], unbiased=False)
        mu = mu.unsqueeze(dim=3)
        var = var.unsqueeze(dim=1)

        # ##### computationally-efficient version of KNormConv based on the paper #########
        weights = self.weight
        sum_w = weights.sum(dim=[1, 2, 3]).unsqueeze(dim=0)
        mu_dot_sum_w = mu * sum_w
        mu_dot_sum_w = mu_dot_sum_w.permute(dims=(0, 3, 1, 2))

        out_conv = (out_conv - mu_dot_sum_w) / torch.sqrt(var + self.eps)

        # apply bias
        if self.bias is not None:
            bias = self.bias
            bias = bias.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
            out_conv = out_conv + bias

        return out_conv

    def __repr__(self) -> str:
        bias = True if self.bias is not None else False
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={bias}, groups={self.groups},  p={self.dropout_p}, eps={self.eps})'

    def __str__(self) -> str:
        return self.__repr__()


class KNConv2d1x1(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout_p: float = 0.05,
                 groups: int = 1,
                 eps: float = 1e-5):
        super(KNConv2d1x1, self).__init__()
        self.kernel_norm1x1 = KernelNorm2d1x1(dropout_p=dropout_p, eps=eps)
        self.conv1x1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True,
                                       dilation=1, groups=groups, padding_mode='zeros', device=None, dtype=None)

    def forward(self, input_t: Tensor) -> Tensor:
        out = self.kernel_norm1x1(input_t)
        out = self.conv1x1(out)
        return out

    def __repr__(self) -> str:
        return f'KNConv2d({self.conv1x1.in_channels}, {self.conv1x1.out_channels}, ' \
               f'kernel_size={self.conv1x1.kernel_size}, stride={self.conv1x1.stride},' \
               f' padding={self.conv1x1.padding}, bias=True, groups={self.conv1x1.groups}, p={self.kernel_norm1x1.dropout_p}, eps={self.kernel_norm1x1.eps})'

    def __str__(self) -> str:
        return self.__repr__()

