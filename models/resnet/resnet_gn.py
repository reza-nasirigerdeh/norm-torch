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
from models.resnet.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


#  ##### GroupNorm versions of ResNet #####
def resnet18_gn(num_classes: int = 1000, num_groups: int = 32, low_resolution: bool = False) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return resnet18(num_classes=num_classes, norm_layer=gn_layer, low_resolution=low_resolution)


def resnet34_gn(num_classes: int = 1000, num_groups: int = 32, low_resolution: bool = False) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return resnet34(num_classes=num_classes, norm_layer=gn_layer, low_resolution=low_resolution)


def resnet50_gn(num_classes: int = 1000, num_groups: int = 32, low_resolution: bool = False) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return resnet50(num_classes=num_classes, norm_layer=gn_layer, low_resolution=low_resolution)


def resnet101_gn(num_classes: int = 1000, num_groups: int = 32, low_resolution: bool = False) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return resnet101(num_classes=num_classes, norm_layer=gn_layer, low_resolution=low_resolution)


def resnet152_gn(num_classes: int = 1000, num_groups: int = 32, low_resolution: bool = False) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return resnet152(num_classes=num_classes, norm_layer=gn_layer, low_resolution=low_resolution)
