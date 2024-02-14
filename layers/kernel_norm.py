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


class KernelNorm1d(torch.nn.Module):
    def __init__(self,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: Union[int, _size_2_t] = 0,
                 dropout_p: float = 0.1,
                 eps: float = 1e-5):
        super(KernelNorm1d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dropout_p = dropout_p
        self.eps = eps

    def forward(self, input_t: Tensor) -> Tensor:
        # apply padding
        x_t = F.pad(input=input_t, pad=self.padding, mode="constant", value=0.0)

        # convert padded-input to sliced-input
        x_t = x_t.unfold(dimension=2, size=self.kernel_size, step=self.stride)
        x_t = x_t.permute(dims=(0, 2, 1, 3))

        # apply dropout with rate dropout_p, and then compute mean and var
        d_x_t = F.dropout(input=x_t, p=self.dropout_p)
        var, mu = torch.var_mean(input=d_x_t, dim=[2, 3], unbiased=False)
        mu = mu.unsqueeze(dim=2).unsqueeze(dim=3)
        var = var.unsqueeze(dim=2).unsqueeze(dim=3)

        # normalize input
        # notice that original input (without dropout) is used as input, but mean and var are
        # computed after applying dropout to the input
        x_t = (x_t - mu) / torch.sqrt(var + self.eps)

        x_t = x_t.permute(dims=(0, 2, 1, 3))
        x_t = x_t.view(x_t.shape[0], x_t.shape[1], -1)

        return x_t

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, p={self.dropout_p}, eps={self.eps})'

    def __str__(self) -> str:
        return self.__repr__()


class KernelNorm2d(torch.nn.Module):
    def __init__(self,
                 kernel_size: Union[int, _size_2_t] = (1, 1),
                 stride: Union[int, _size_2_t] = (1, 1),
                 padding: Union[int, _size_2_t, _size_4_t] = (0, 0),
                 dropout_p: float = 0.1,
                 eps: float = 1e-5):
        super(KernelNorm2d, self).__init__()

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding, padding) if isinstance(padding, int) else padding
        self.padding = self.padding if len(self.padding) == 4 else (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        self.dropout_p = dropout_p
        self.eps = eps

    def forward(self, input_t: Tensor) -> Tensor:
        # add padding
        x_t = F.pad(input=input_t, pad=self.padding, mode="constant", value=0.0)

        # convert padded-input to sliced-input
        x_t = x_t.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        x_t = x_t.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])

        x_t = x_t.permute(dims=(0, 2, 3, 1, 4, 5))

        # apply dropout with rate dropout_p, and then compute mean and var
        d_x_t = F.dropout(input=x_t, p=self.dropout_p)
        var, mu = torch.var_mean(input=d_x_t, dim=[3, 4, 5], unbiased=False)
        mu = mu.unsqueeze(dim=3).unsqueeze(dim=4).unsqueeze(dim=5)
        var = var.unsqueeze(dim=3).unsqueeze(dim=4).unsqueeze(dim=5)

        # normalize input
        # notice that original input (without dropout) is used as input, but mean and var are
        # computed after applying dropout to the input
        x_t = (x_t - mu) / torch.sqrt(var + self.eps)

        x_t = x_t.permute(dims=(0, 3, 1, 4, 2, 5))
        x_t = x_t.view(x_t.shape[0], x_t.shape[1], -1, x_t.shape[-2], x_t.shape[-1])
        x_t = x_t.view(x_t.shape[0], x_t.shape[1], x_t.shape[2], -1)

        return x_t

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, p={self.dropout_p}, eps={self.eps})'

    def __str__(self) -> str:
        return self.__repr__()


class KernelNorm2d1x1(torch.nn.Module):
    """ Computationally-efficient version of KernelNorm2d in the case, where kernel_size=(1,1), stride=(1,1), and padding=(0,0) """
    def __init__(self, dropout_p: float = 0.1, eps: float = 1e-5):
        super(KernelNorm2d1x1, self).__init__()
        self.dropout_p = dropout_p
        self.eps = eps

    def forward(self, input_t: Tensor) -> Tensor:
        d_x_t = F.dropout(input=input_t, p=self.dropout_p)
        var, mu = torch.var_mean(input=d_x_t, dim=[1], keepdim=True, unbiased=False)
        x_t = (input_t - mu) / torch.sqrt(var + self.eps)

        return x_t

    def __repr__(self) -> str:
        return f'KernelNorm2d(kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), p={self.dropout_p}, eps={self.eps})'

    def __str__(self) -> str:
        return self.__repr__()
