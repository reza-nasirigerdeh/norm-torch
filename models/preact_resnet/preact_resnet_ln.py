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

from torch import nn
from models.preact_resnet.preact_resnet import preact_resnet18, preact_resnet34, preact_resnet50, preact_resnet101, preact_resnet152

# LayerNorm is equivalent to GroupNorm with number of groups of 1
ln_layer = lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels)


#  ##### LayerNorm versions of PreactResNet #####
def preact_resnet18_ln(num_classes: int = 1000, low_resolution: bool = False) -> nn.Module:
    return preact_resnet18(num_classes=num_classes, norm_layer=ln_layer, low_resolution=low_resolution)


def preact_resnet34_ln(num_classes: int = 1000, low_resolution: bool = False) -> nn.Module:
    return preact_resnet34(num_classes=num_classes, norm_layer=ln_layer, low_resolution=low_resolution)


def preact_resnet50_ln(num_classes: int = 1000, low_resolution: bool = False) -> nn.Module:
    return preact_resnet50(num_classes=num_classes, norm_layer=ln_layer, low_resolution=low_resolution)


def preact_resnet101_ln(num_classes: int = 1000, low_resolution: bool = False) -> nn.Module:
    return preact_resnet101(num_classes=num_classes, norm_layer=ln_layer, low_resolution=low_resolution)


def preact_resnet152_ln(num_classes: int = 1000, low_resolution: bool = False) -> nn.Module:
    return preact_resnet152(num_classes=num_classes, norm_layer=ln_layer, low_resolution=low_resolution)
