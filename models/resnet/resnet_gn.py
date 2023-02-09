"""
    Copyright 2023 Reza NasiriGerdeh. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from torchvision import models
from torch import nn


#  ##### GroupNorm versions of ResNet #####
def resnet18_gn(num_classes=10, group_size=32):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return models.resnet18(num_classes=num_classes, norm_layer=gn_layer)


def resnet34_gn(num_classes=10, group_size=32):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return models.resnet34(num_classes=num_classes, norm_layer=gn_layer)


def resnet50_gn(num_classes=10, group_size=32):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return models.resnet50(num_classes=num_classes, norm_layer=gn_layer)


def resnet101_gn(num_classes=10, group_size=32):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return models.resnet101(num_classes=num_classes, norm_layer=gn_layer)


def resnet152_gn(num_classes=10, group_size=32):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return models.resnet152(num_classes=num_classes, norm_layer=gn_layer)
