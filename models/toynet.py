"""
    Simple fully-connect and convolutional networks for testing purposes.
    These models only work on the MNIST/Fashion-MNIST dataset.

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


class FNN(nn.Module):
    """ A simple fully-connected network, which only works on the MNIST/Fashion-MNIST dataset """
    def __init__(self, num_classes):
        super(FNN, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features=784, out_features=64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=64, out_features=128), nn.ReLU())
        self.classifier = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = torch.flatten(input=x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x


class CNN(nn.Module):
    """ A simple convolutional network, which only works on the MNIST/Fashion-MNIST dataset """
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.avg_pool(x)
        x = torch.flatten(input=x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x
