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

import torch
from torch import nn


def _make_vgg_block(in_channels, out_channels, conv_layer, norm_layer, kn_dropout_p, max_pooling=False, activation=nn.ReLU):

    layers = list()
    bias = True if norm_layer is None else False
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), bias=bias))

    if norm_layer is not None:
        layers.append(norm_layer(out_channels))

    layers.append(activation())

    if max_pooling:
        layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

    return nn.Sequential(*layers)


class VGG6(nn.Module):
    def __init__(self, num_classes, conv_layer, norm_layer, kn_dropout_p=0.1, activation=nn.ReLU):
        super(VGG6, self).__init__()

        self.block1 = _make_vgg_block(in_channels=3, out_channels=64, conv_layer=conv_layer, norm_layer=norm_layer,
                                      kn_dropout_p=kn_dropout_p, max_pooling=True, activation=activation)

        self.block2 = _make_vgg_block(in_channels=64, out_channels=128, conv_layer=conv_layer, norm_layer=norm_layer,
                                      kn_dropout_p=kn_dropout_p, max_pooling=True, activation=activation)

        self.block3 = _make_vgg_block(in_channels=128, out_channels=256, conv_layer=conv_layer, norm_layer=norm_layer,
                                      kn_dropout_p=kn_dropout_p, max_pooling=True, activation=activation)

        self.block4 = _make_vgg_block(in_channels=256, out_channels=512, conv_layer=conv_layer, norm_layer=norm_layer,
                                      kn_dropout_p=kn_dropout_p, max_pooling=False, activation=activation)

        self.block5 = _make_vgg_block(in_channels=512, out_channels=512, conv_layer=conv_layer, norm_layer=norm_layer,
                                      kn_dropout_p=kn_dropout_p, max_pooling=False, activation=activation)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = torch.flatten(input=x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x


def vgg6_nn(num_classes=10, activation=nn.ReLU):
    return VGG6(num_classes=num_classes, conv_layer=nn.Conv2d, norm_layer=None, activation=activation)


def vgg6_bn(num_classes=10, activation=nn.ReLU):
    return VGG6(num_classes=num_classes, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, activation=activation)


def vgg6_ln(num_classes=10, activation=nn.ReLU):
    ln_layer = lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return VGG6(num_classes=num_classes, conv_layer=nn.Conv2d, norm_layer=ln_layer, activation=activation)


def vgg6_gn(num_classes=10, group_size=32, activation=nn.ReLU):
    gn_layer = lambda num_channels: nn.GroupNorm(num_groups=num_channels//group_size, num_channels=num_channels)
    return VGG6(num_classes=num_classes, conv_layer=nn.Conv2d, norm_layer=gn_layer, activation=activation)
