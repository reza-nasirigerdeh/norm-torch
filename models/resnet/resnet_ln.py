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

# LayerNorm is equivalent to GroupNorm with number of groups of 1
ln_layer = lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels)


#  ##### LayerNorm versions of ResNet #####
def resnet18_ln(num_classes=10):
    return models.resnet18(num_classes=num_classes, norm_layer=ln_layer)


def resnet34_ln(num_classes=10):
    return models.resnet34(num_classes=num_classes, norm_layer=ln_layer)


def resnet50_ln(num_classes=10):
    return models.resnet50(num_classes=num_classes, norm_layer=ln_layer)


def resnet101_ln(num_classes=10):
    return models.resnet101(num_classes=num_classes, norm_layer=ln_layer)


def resnet152_ln(num_classes=10):
    return models.resnet152(num_classes=num_classes, norm_layer=ln_layer)
