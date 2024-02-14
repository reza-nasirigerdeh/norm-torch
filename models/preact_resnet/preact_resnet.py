
# From : https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py,
# Modifications: (1) the low_resolution boolean argument was added to the PreactResNets to be trained on low-resolution datasets (e.g. CIFAR)
#                (2) the normalization layer was added as additional argument to have batch/layer/group/no normalized versions of PreactResNets.


import torch.nn as nn
import torch


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(PreactBasicBlock, self).__init__()
        self.norm1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.norm2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=(1, 1), stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.relu2(self.norm2(out))
        out = self.conv2(out)
        out += shortcut
        return out


class PreactBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(PreactBottleneckBlock, self).__init__()
        self.norm1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.norm2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.norm3 = norm_layer(planes)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=(1, 1), bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=(1, 1), stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.relu2(self.norm2(out))
        out = self.conv2(out)
        out = self.relu3(self.norm3(out))
        out = self.conv3(out)
        out += shortcut
        return out


class PreactResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=nn.BatchNorm2d, low_resolution=False):
        super(PreactResNet, self).__init__()
        self.in_planes = 64

        if low_resolution:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(self.in_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_layer=norm_layer)

        self.norm_f = norm_layer(512*block.expansion)
        self.relu_f = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.norm_f(out)
        out = self.relu_f(out)

        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


def preact_resnet18(num_classes=1000, norm_layer=nn.BatchNorm2d, low_resolution=False):
    return PreactResNet(PreactBasicBlock, [2, 2, 2, 2], num_classes, norm_layer, low_resolution)


def preact_resnet34(num_classes=1000, norm_layer=nn.BatchNorm2d, low_resolution=False):
    return PreactResNet(PreactBasicBlock, [3, 4, 6, 3], num_classes, norm_layer, low_resolution)


def preact_resnet50(num_classes=1000, norm_layer=nn.BatchNorm2d, low_resolution=False):
    return PreactResNet(PreactBottleneckBlock, [3, 4, 6, 3], num_classes, norm_layer, low_resolution)


def preact_resnet101(num_classes=1000, norm_layer=nn.BatchNorm2d, low_resolution=False):
    return PreactResNet(PreactBottleneckBlock, [3, 4, 23, 3], num_classes, norm_layer, low_resolution)


def preact_resnet152(num_classes=1000, norm_layer=nn.BatchNorm2d, low_resolution=False):
    return PreactResNet(PreactBottleneckBlock, [3, 8, 36, 3], num_classes, norm_layer, low_resolution)
