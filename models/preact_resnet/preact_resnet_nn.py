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

from torch import nn
from models.preact_resnet.preact_resnet import preact_resnet18, preact_resnet34, preact_resnet50, preact_resnet101, preact_resnet152


#  ##### PreactResNet models with Identity as norm layer #####
def preact_resnet18_nn(num_classes=10):
    return preact_resnet18(num_classes=num_classes, norm_layer=nn.Identity)


def preact_resnet34_nn(num_classes=10):
    return preact_resnet34(num_classes=num_classes, norm_layer=nn.Identity)


def preact_resnet50_nn(num_classes=10):
    return preact_resnet50(num_classes=num_classes, norm_layer=nn.Identity)


def preact_resnet101_nn(num_classes=10):
    return preact_resnet101(num_classes=num_classes, norm_layer=nn.Identity)


def preact_resnet152_nn(num_classes=10):
    return preact_resnet152(num_classes=num_classes, norm_layer=nn.Identity)
