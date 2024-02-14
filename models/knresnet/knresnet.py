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
from torch import nn
from torch import Tensor
from layers.knconv import KNConv2d, KNConv2d1x1
from layers.kernel_norm import KernelNorm2d1x1


class ResidualBlockKN(nn.Module):
    def __init__(self, channels: int, inter_channels: int, dropout_p: float = 0.05):
        super(ResidualBlockKN, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = KNConv2d(channels, inter_channels, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=dropout_p)

        self.relu2 = nn.ReLU()
        self.conv2 = KNConv2d(inter_channels, channels, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=True, dropout_p=dropout_p)

    def forward(self, input_t: Tensor) -> Tensor:
        identity = input_t
        out = self.relu1(input_t)
        out = self.conv1(out)

        out = self.relu2(out)
        out = self.conv2(out)

        out += identity
        return out


class TransBlockKN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: tuple = (0, 0, 0, 0), dropout_p: float = 0.05):
        super(TransBlockKN, self).__init__()
        self.relu = nn.ReLU()
        self.conv = KNConv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(1, 1), padding=padding, bias=True, dropout_p=dropout_p)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    def forward(self, input_t: Tensor) -> Tensor:
        out = self.relu(input_t)
        out = self.conv(out)
        out = self.max_pool(out)

        return out


class ConvBlockKN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: tuple, dropout_p: float = 0.05):
        super(ConvBlockKN, self).__init__()
        self.relu = nn.ReLU()
        self.conv = KNConv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(1, 1), padding=padding, bias=True, dropout_p=dropout_p)

    def forward(self, input_t: Tensor) -> Tensor:
        out = self.relu(input_t)
        out = self.conv(out)

        return out


class ResidualBottleneckKN(nn.Module):
    def __init__(self, channels: int, inter_channels: int, dropout_p: float = 0.05):
        super(ResidualBottleneckKN, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = KNConv2d(channels, inter_channels, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=dropout_p)

        self.relu2 = nn.ReLU()
        self.conv2 = KNConv2d(inter_channels, inter_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=dropout_p)

        self.relu3 = nn.ReLU()
        self.conv3 = KNConv2d(inter_channels, channels, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=True, dropout_p=dropout_p)

    def forward(self, input_t: Tensor) -> Tensor:
        identity = input_t
        out = self.relu1(input_t)
        out = self.conv1(out)

        out = self.relu2(out)
        out = self.conv2(out)

        out = self.relu3(out)
        out = self.conv3(out)

        out += identity
        return out


class TransBlockKN1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.05):
        super(TransBlockKN1x1, self).__init__()
        self.relu = nn.ReLU()
        self.conv = KNConv2d1x1(in_channels, out_channels, dropout_p=dropout_p)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    def forward(self, input_t):
        out = self.relu(input_t)
        out = self.conv(out)
        out = self.max_pool(out)

        return out


class ResidualBottleneckKN1x1(nn.Module):
    def __init__(self, channels: int, inter_channels: int, dropout_p: float = 0.05):
        super(ResidualBottleneckKN1x1, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = KNConv2d1x1(channels, inter_channels, dropout_p=dropout_p)

        self.relu2 = nn.ReLU()
        self.conv2 = KNConv2d(inter_channels, inter_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=dropout_p)

        self.relu3 = nn.ReLU()
        self.conv3 = KNConv2d1x1(inter_channels, channels, dropout_p=dropout_p)

    def forward(self, input_t):
        identity = input_t
        out = self.relu1(input_t)
        out = self.conv1(out)

        out = self.relu2(out)
        out = self.conv2(out)

        out = self.relu3(out)
        out = self.conv3(out)

        out += identity
        return out


class KNResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout_p: float = 0.05, low_resolution: bool = False):

        """ 18-layer kernel normalized residual network """
        super(KNResNet18, self).__init__()

        if low_resolution:
            self.block0 = nn.Sequential(
                KNConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=dropout_p)
            )
        else:
            self.block0 = nn.Sequential(
                KNConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True, dropout_p=dropout_p),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )

        self.res_block1 = ResidualBlockKN(channels=64, inter_channels=256, dropout_p=dropout_p)
        self.res_block2 = ResidualBlockKN(channels=64, inter_channels=256, dropout_p=dropout_p)

        self.trans_block1 = TransBlockKN(in_channels=64, out_channels=256, padding=(1, 0, 1, 0), dropout_p=dropout_p)
        self.res_block3 = ResidualBlockKN(channels=256, inter_channels=256, dropout_p=dropout_p)
        self.res_block4 = ResidualBlockKN(channels=256, inter_channels=256, dropout_p=dropout_p)

        self.trans_block2 = TransBlockKN(in_channels=256, out_channels=512, padding=(0, 1, 0, 1), dropout_p=dropout_p)
        self.res_block5 = ResidualBlockKN(channels=512, inter_channels=512, dropout_p=dropout_p)

        padding = (1, 0, 1, 0) if low_resolution else (2, 1, 2, 1)
        self.trans_block3 = TransBlockKN(in_channels=512, out_channels=724, padding=padding, dropout_p=dropout_p)
        self.res_block6 = ResidualBlockKN(channels=724, inter_channels=724, dropout_p=dropout_p)

        self.max_pool_f = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        padding = (1, 1) if low_resolution else (0, 0)
        self.conv_block_f = ConvBlockKN(in_channels=724, out_channels=512, padding=padding, dropout_p=dropout_p)

        kn_dropout_p = min(5 * dropout_p, 0.25)
        self.kn_f = KernelNorm2d1x1(dropout_p=kn_dropout_p)
        self.relu_f = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, KNConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input_t: Tensor) -> Tensor:
        out = self.block0(input_t)

        out = self.res_block1(out)
        out = self.res_block2(out)

        out = self.trans_block1(out)
        out = self.res_block3(out)
        out = self.res_block4(out)

        out = self.trans_block2(out)
        out = self.res_block5(out)

        out = self.trans_block3(out)
        out = self.res_block6(out)

        out = self.max_pool_f(out)
        out = self.conv_block_f(out)

        out = self.kn_f(out)
        out = self.relu_f(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


class KNResNet34(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout_p: float = 0.05, low_resolution: bool = False):

        """ 34-layer kernel normalized residual network """
        super(KNResNet34, self).__init__()

        if low_resolution:
            self.block0 = nn.Sequential(
                KNConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=dropout_p)
            )
        else:
            self.block0 = nn.Sequential(
                KNConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True, dropout_p=dropout_p),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )

        self.res_block1 = ResidualBlockKN(channels=64, inter_channels=256, dropout_p=dropout_p)
        self.res_block2 = ResidualBlockKN(channels=64, inter_channels=256, dropout_p=dropout_p)
        self.res_block3 = ResidualBlockKN(channels=64, inter_channels=256, dropout_p=dropout_p)
        self.res_block4 = ResidualBlockKN(channels=64, inter_channels=256, dropout_p=dropout_p)

        self.trans_block1 = TransBlockKN(in_channels=64, out_channels=256, padding=(1, 0, 1, 0), dropout_p=dropout_p)
        self.res_block5 = ResidualBlockKN(channels=256, inter_channels=320, dropout_p=dropout_p)
        self.res_block6 = ResidualBlockKN(channels=256, inter_channels=320, dropout_p=dropout_p)
        self.res_block7 = ResidualBlockKN(channels=256, inter_channels=320, dropout_p=dropout_p)
        self.res_block8 = ResidualBlockKN(channels=256, inter_channels=320, dropout_p=dropout_p)
        self.res_block9 = ResidualBlockKN(channels=256, inter_channels=320, dropout_p=dropout_p)

        self.trans_block2 = TransBlockKN(in_channels=256, out_channels=512, padding=(0, 1, 0, 1), dropout_p=dropout_p)
        self.res_block10 = ResidualBlockKN(channels=512, inter_channels=640, dropout_p=dropout_p)
        self.res_block11 = ResidualBlockKN(channels=512, inter_channels=640, dropout_p=dropout_p)
        self.res_block12 = ResidualBlockKN(channels=512, inter_channels=640, dropout_p=dropout_p)

        padding = (1, 0, 1, 0) if low_resolution else (2, 1, 2, 1)
        self.trans_block3 = TransBlockKN(in_channels=512, out_channels=512, padding=padding, dropout_p=dropout_p)
        self.res_block13 = ResidualBlockKN(channels=512, inter_channels=843, dropout_p=dropout_p)
        self.res_block14 = ResidualBlockKN(channels=512, inter_channels=843, dropout_p=dropout_p)

        self.max_pool_f = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        padding = (1, 1) if low_resolution else (0, 0)
        self.conv_block_f = ConvBlockKN(in_channels=512, out_channels=512, padding=padding, dropout_p=dropout_p)

        kn_dropout_p = min(5 * dropout_p, 0.25)
        self.kn_f = KernelNorm2d1x1(dropout_p=kn_dropout_p)
        self.relu_f = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, KNConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input_t: Tensor) -> Tensor:
        out = self.block0(input_t)

        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)

        out = self.trans_block1(out)
        out = self.res_block5(out)
        out = self.res_block6(out)
        out = self.res_block7(out)
        out = self.res_block8(out)
        out = self.res_block9(out)

        out = self.trans_block2(out)
        out = self.res_block10(out)
        out = self.res_block11(out)
        out = self.res_block12(out)

        out = self.trans_block3(out)
        out = self.res_block13(out)
        out = self.res_block14(out)

        out = self.max_pool_f(out)
        out = self.conv_block_f(out)

        out = self.kn_f(out)
        out = self.relu_f(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


class KNResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout_p: float = 0.05, low_resolution: bool = False):

        """ 50-layer kernel normalized residual network """
        super(KNResNet50, self).__init__()

        if low_resolution:
            self.block0 = nn.Sequential(
                KNConv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=dropout_p)
            )
        else:
            self.block0 = nn.Sequential(
                KNConv2d(3, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True, dropout_p=dropout_p),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )

        self.res_block1 = ResidualBottleneckKN(channels=256, inter_channels=64, dropout_p=dropout_p)
        self.res_block2 = ResidualBottleneckKN(channels=256, inter_channels=64, dropout_p=dropout_p)
        self.res_block3 = ResidualBottleneckKN(channels=256, inter_channels=64, dropout_p=dropout_p)
        self.res_block4 = ResidualBottleneckKN(channels=256, inter_channels=64, dropout_p=dropout_p)

        self.trans_block1 = TransBlockKN(in_channels=256, out_channels=512, padding=(1, 0, 1, 0), dropout_p=dropout_p)
        self.res_block5 = ResidualBottleneckKN(channels=512, inter_channels=128, dropout_p=dropout_p)
        self.res_block6 = ResidualBottleneckKN(channels=512, inter_channels=128, dropout_p=dropout_p)
        self.res_block7 = ResidualBottleneckKN(channels=512, inter_channels=128, dropout_p=dropout_p)
        self.res_block8 = ResidualBottleneckKN(channels=512, inter_channels=128, dropout_p=dropout_p)
        self.res_block9 = ResidualBottleneckKN(channels=512, inter_channels=128, dropout_p=dropout_p)

        self.trans_block2 = TransBlockKN(in_channels=512, out_channels=810, padding=(0, 1, 0, 1), dropout_p=dropout_p)
        self.res_block10 = ResidualBottleneckKN(channels=810, inter_channels=201, dropout_p=dropout_p)
        self.res_block11 = ResidualBottleneckKN(channels=810, inter_channels=201, dropout_p=dropout_p)
        self.res_block12 = ResidualBottleneckKN(channels=810, inter_channels=201, dropout_p=dropout_p)
        self.res_block13 = ResidualBottleneckKN(channels=810, inter_channels=201, dropout_p=dropout_p)

        self.trans_block3 = TransBlockKN1x1(in_channels=810, out_channels=2048, dropout_p=dropout_p)
        self.res_block14 = ResidualBottleneckKN1x1(channels=2048, inter_channels=512, dropout_p=dropout_p)
        self.res_block15 = ResidualBottleneckKN1x1(channels=2048, inter_channels=512, dropout_p=dropout_p)

        kn_dropout_p = min(5 * dropout_p, 0.25)
        self.kn_f = KernelNorm2d1x1(dropout_p=kn_dropout_p)
        self.relu_f = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, KNConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, KNConv2d1x1):
                nn.init.kaiming_normal_(m.conv1x1.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.conv1x1.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input_t: Tensor) -> Tensor:
        out = self.block0(input_t)

        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)

        out = self.trans_block1(out)
        out = self.res_block5(out)
        out = self.res_block6(out)
        out = self.res_block7(out)
        out = self.res_block8(out)
        out = self.res_block9(out)

        out = self.trans_block2(out)
        out = self.res_block10(out)
        out = self.res_block11(out)
        out = self.res_block12(out)
        out = self.res_block13(out)

        out = self.trans_block3(out)
        out = self.res_block14(out)
        out = self.res_block15(out)

        out = self.kn_f(out)
        out = self.relu_f(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


def knresnet18(num_classes: int = 1000, dropout_p: float = 0.05, low_resolution: bool = False):
    return KNResNet18(num_classes, dropout_p, low_resolution)


def knresnet34(num_classes: int = 1000, dropout_p: float = 0.05, low_resolution: bool = False):
    return KNResNet34(num_classes, dropout_p, low_resolution)


def knresnet50(num_classes: int = 1000, dropout_p: float = 0.05, low_resolution: bool = False):
    return KNResNet50(num_classes, dropout_p, low_resolution)

